package com.demo.aiknowledge.service.impl;

import com.demo.aiknowledge.dto.*;
import com.demo.aiknowledge.entity.*;
import com.demo.aiknowledge.mapper.*;
import com.demo.aiknowledge.service.KnowledgeInspectionService;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDateTime;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.Comparator;

@Service
@Slf4j
public class KnowledgeInspectionServiceImpl implements KnowledgeInspectionService {

    @Resource
    private QaUnansweredMapper qaUnansweredMapper;

    @Resource
    private KnowledgeCategoryMapper knowledgeCategoryMapper;

    @Resource
    private KnowledgeDocMapper knowledgeDocMapper;

    @Resource
    private KnowledgeChunkMapper knowledgeChunkMapper;

    @Resource
    private DocViewLogMapper docViewLogMapper;

    @Resource
    private RestTemplate restTemplate;

    @Value("${ai.service.url}")
    private String aiServiceUrl;

    private static final Pattern CHINESE_PATTERN = Pattern.compile("[\\u4e00-\\u9fa5]+");
    private static final Pattern KEYWORD_PATTERN = Pattern.compile("[\\u4e00-\\u9fa5]{2,}|[a-zA-Z]+");
    private static final int DEFAULT_MIN_CHUNK_LENGTH = 50;
    private static final int DEFAULT_MAX_CHUNK_LENGTH = 5000;
    private static final int DEFAULT_OUTDATED_DAYS = 30;
    private static final int DEFAULT_UNACCESSED_DAYS = 7;

    @Override
    public UnansweredAnalysisResponse analyzeUnansweredQuestions(UnansweredAnalysisRequest request) {
        List<QaUnanswered> allRecords = qaUnansweredMapper.selectAll();

        if (request.getStartDate() != null || request.getEndDate() != null) {
            LocalDateTime start = request.getStartDate() != null ?
                    request.getStartDate().atStartOfDay() : null;
            LocalDateTime end = request.getEndDate() != null ?
                    request.getEndDate().atTime(LocalTime.MAX) : null;
            allRecords = filterByTimeRange(allRecords, start, end);
        }

        if (request.getMinCount() != null && request.getMinCount() > 1) {
            allRecords = allRecords.stream()
                    .filter(r -> r.getCount() >= request.getMinCount())
                    .collect(Collectors.toList());
        }

        List<UnansweredCluster> clusters = semanticClusterOrFallback(allRecords, request.getClusterThreshold());
        List<KnowledgeGapSuggestion> suggestions = generateSuggestions(clusters);
        List<QaUnansweredExport> exportData = convertToExportData(allRecords);

        UnansweredAnalysisResponse response = new UnansweredAnalysisResponse();
        response.setTotalUnansweredCount(allRecords.stream().mapToInt(QaUnanswered::getCount).sum());
        response.setTotalUniqueQuestions(allRecords.size());
        response.setClusterCount(clusters.size());
        response.setClusters(clusters);
        response.setSuggestions(suggestions);
        response.setExportData(exportData);

        return response;
    }

    private List<UnansweredCluster> semanticClusterOrFallback(List<QaUnanswered> records, double threshold) {
        try {
            List<Map<String, Object>> semanticClusters = callSemanticCluster(records, threshold);
            if (semanticClusters != null && !semanticClusters.isEmpty()) {
                return convertSemanticClusters(semanticClusters);
            }
        } catch (Exception e) {
            log.warn("Semantic clustering failed, fallback to character clustering: {}", e.getMessage());
        }
        return clusterQuestions(records, threshold);
    }

    private List<Map<String, Object>> callSemanticCluster(List<QaUnanswered> records, double threshold) {
        List<Map<String, Object>> questions = records.stream()
                .map(r -> {
                    Map<String, Object> q = new HashMap<>();
                    q.put("question", r.getQuestion());
                    q.put("count", r.getCount());
                    return q;
                })
                .collect(Collectors.toList());

        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("questions", questions);
        requestBody.put("threshold", threshold);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(requestBody, headers);

        ResponseEntity<Map> response = restTemplate.postForEntity(
                aiServiceUrl + "/knowledge-gap/semantic-cluster", entity, Map.class);

        if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
            Object clustersObj = response.getBody().get("clusters");
            if (clustersObj instanceof List) {
                return (List<Map<String, Object>>) clustersObj;
            }
        }
        return null;
    }

    private List<UnansweredCluster> convertSemanticClusters(List<Map<String, Object>> semanticClusters) {
        return semanticClusters.stream()
                .map(sc -> {
                    UnansweredCluster cluster = new UnansweredCluster();
                    cluster.setTopic(String.valueOf(sc.getOrDefault("topic", "未分类")));
                    String summary = sc.get("summary") != null
                            ? String.valueOf(sc.get("summary"))
                            : cluster.getTopic();
                    cluster.setTopicSummary(summary);
                    cluster.setTotalCount(((Number) sc.getOrDefault("total_count", 0)).intValue());
                    Object questionsObj = sc.get("questions");
                    if (questionsObj instanceof List) {
                        cluster.setQuestions((List<String>) questionsObj);
                    } else {
                        cluster.setQuestions(new ArrayList<>());
                    }
                    Object keywordsObj = sc.get("keywords");
                    List<String> keywords = new ArrayList<>();
                    if (keywordsObj instanceof List) {
                        for (Object kw : (List<?>) keywordsObj) {
                            keywords.add(String.valueOf(kw));
                        }
                    }
                    cluster.setSuggestedKeywords(keywords);
                    return cluster;
                })
                .collect(Collectors.toList());
    }

    @Override
    public Map<String, Object> getUnansweredStatistics() {
        List<QaUnanswered> allRecords = qaUnansweredMapper.selectAll();

        Map<String, Object> stats = new HashMap<>();
        stats.put("totalUnansweredCount", allRecords.stream().mapToInt(QaUnanswered::getCount).sum());
        stats.put("totalUniqueQuestions", allRecords.size());
        stats.put("topQuestions", allRecords.stream()
                .sorted((a, b) -> Integer.compare(b.getCount(), a.getCount()))
                .limit(10)
                .map(r -> Map.of("question", r.getQuestion(), "count", r.getCount()))
                .collect(Collectors.toList()));

        return stats;
    }

    private List<QaUnanswered> filterByTimeRange(List<QaUnanswered> records, LocalDateTime start, LocalDateTime end) {
        return records.stream().filter(r -> {
            LocalDateTime createTime = r.getCreateTime();
            if (createTime == null) {
                return false;  // createTime 为 null 无法判断，过滤掉
            }
            return (start == null || !createTime.isBefore(start)) && (end == null || !createTime.isAfter(end));
        }).collect(Collectors.toList());
    }

    private List<UnansweredCluster> clusterQuestions(List<QaUnanswered> records, double threshold) {
        if (records.isEmpty()) return Collections.emptyList();

        Map<String, List<QaUnanswered>> clusters = new HashMap<>();
        Set<QaUnanswered> processed = new HashSet<>();

        for (QaUnanswered record : records) {
            if (processed.contains(record)) continue;

            String seedQuestion = record.getQuestion();

            List<QaUnanswered> cluster = new ArrayList<>();
            cluster.add(record);
            processed.add(record);

            for (QaUnanswered other : records) {
                if (processed.contains(other)) continue;
                if (isSimilar(seedQuestion, other.getQuestion(), threshold)) {
                    cluster.add(other);
                    processed.add(other);
                }
            }

            if (!cluster.isEmpty()) {
                String topic = generateClusterTopic(cluster);
                clusters.put(topic, cluster);
            }
        }

        return clusters.entrySet().stream()
                .map(entry -> {
                    UnansweredCluster cluster = new UnansweredCluster();
                    cluster.setTopic(entry.getKey());
                    cluster.setTopicSummary(summarizeTopic(entry.getValue()));
                    cluster.setTotalCount(entry.getValue().stream().mapToInt(QaUnanswered::getCount).sum());
                    cluster.setQuestions(entry.getValue().stream()
                            .map(QaUnanswered::getQuestion)
                            .collect(Collectors.toList()));
                    cluster.setSuggestedKeywords(extractKeywords(entry.getValue()));
                    return cluster;
                })
                .sorted((a, b) -> Integer.compare(b.getTotalCount(), a.getTotalCount()))
                .collect(Collectors.toList());
    }

    private boolean isSimilar(String q1, String q2, double threshold) {
        String clean1 = cleanQuestion(q1);
        String clean2 = cleanQuestion(q2);

        int commonChars = countCommonCharacters(clean1, clean2);
        int maxLen = Math.max(clean1.length(), clean2.length());

        return maxLen > 0 && (double) commonChars / maxLen >= threshold;
    }

    private String cleanQuestion(String question) {
        if (question == null) return "";
        return question.toLowerCase()
                .replaceAll("[\\s\\p{Punct}]", "");
    }

    private int countCommonCharacters(String s1, String s2) {
        Set<Character> chars1 = s1.chars().mapToObj(c -> (char) c).collect(Collectors.toSet());
        return (int) s2.chars().filter(c -> chars1.contains((char) c)).count();
    }

    private String generateClusterTopic(List<QaUnanswered> cluster) {
        Map<String, Integer> wordCount = new HashMap<>();

        for (QaUnanswered record : cluster) {
            Matcher matcher = KEYWORD_PATTERN.matcher(record.getQuestion());
            while (matcher.find()) {
                String word = matcher.group();
                wordCount.put(word, wordCount.getOrDefault(word, 0) + record.getCount());
            }
        }

        return wordCount.entrySet().stream()
                .sorted((a, b) -> Integer.compare(b.getValue(), a.getValue()))
                .limit(3)
                .map(Map.Entry::getKey)
                .collect(Collectors.joining("、"));
    }

    private String summarizeTopic(List<QaUnanswered> cluster) {
        if (cluster.isEmpty()) return "";

        QaUnanswered top = cluster.stream()
                .max(Comparator.comparingInt(QaUnanswered::getCount))
                .orElse(cluster.get(0));

        String question = top.getQuestion();
        if (question.length() <= 20) return question;
        return question.substring(0, 20) + "...";
    }

    private List<String> extractKeywords(List<QaUnanswered> cluster) {
        Map<String, Integer> wordCount = new HashMap<>();

        for (QaUnanswered record : cluster) {
            Matcher matcher = KEYWORD_PATTERN.matcher(record.getQuestion());
            while (matcher.find()) {
                String word = matcher.group();
                if (word.length() >= 2) {
                    wordCount.put(word, wordCount.getOrDefault(word, 0) + record.getCount());
                }
            }
        }

        return wordCount.entrySet().stream()
                .sorted((a, b) -> Integer.compare(b.getValue(), a.getValue()))
                .limit(5)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
    }

    private List<KnowledgeGapSuggestion> generateSuggestions(List<UnansweredCluster> clusters) {
        List<KnowledgeCategory> categories = knowledgeCategoryMapper.selectList(null);
        List<String> existingCategories = categories.stream()
                .map(KnowledgeCategory::getName)
                .collect(Collectors.toList());

        List<KnowledgeGapSuggestion> suggestions = new ArrayList<>();

        for (UnansweredCluster cluster : clusters) {
            KnowledgeGapSuggestion suggestion = new KnowledgeGapSuggestion();
            suggestion.setTopic(cluster.getTopic());
            suggestion.setQuestionCount(cluster.getTotalCount());
            suggestion.setSuggestedKeywords(cluster.getSuggestedKeywords());

            String category = matchCategory(cluster, existingCategories);
            suggestion.setRelatedCategory(category);

            if (cluster.getTotalCount() >= 10) {
                suggestion.setPriority("高");
                suggestion.setSuggestionType("紧急补库");
                suggestion.setSuggestion("该主题问题频繁出现，建议立即补充相关知识库文档");
            } else if (cluster.getTotalCount() >= 5) {
                suggestion.setPriority("中");
                suggestion.setSuggestionType("建议补库");
                suggestion.setSuggestion("该主题存在知识缺口，建议补充相关文档");
            } else {
                suggestion.setPriority("低");
                suggestion.setSuggestionType("观察");
                suggestion.setSuggestion("该主题问题较少，建议继续观察");
            }

            suggestions.add(suggestion);
        }

        return suggestions.stream()
                .sorted((a, b) -> {
                    int priorityOrder = getPriorityOrder(b.getPriority()) - getPriorityOrder(a.getPriority());
                    return priorityOrder != 0 ? priorityOrder : Integer.compare(b.getQuestionCount(), a.getQuestionCount());
                })
                .collect(Collectors.toList());
    }

    private String matchCategory(UnansweredCluster cluster, List<String> categories) {
        for (String keyword : cluster.getSuggestedKeywords()) {
            for (String category : categories) {
                if (category.contains(keyword) || keyword.contains(category)) {
                    return category;
                }
            }
        }
        return "未分类";
    }

    private int getPriorityOrder(String priority) {
        return switch (priority) {
            case "高" -> 3;
            case "中" -> 2;
            case "低" -> 1;
            default -> 0;
        };
    }

    private List<QaUnansweredExport> convertToExportData(List<QaUnanswered> records) {
        return records.stream()
                .map(r -> {
                    QaUnansweredExport export = new QaUnansweredExport();
                    export.setQuestion(r.getQuestion());
                    export.setCount(r.getCount());
                    export.setFirstOccurrence(r.getCreateTime());
                    export.setLastOccurrence(r.getUpdateTime());
                    return export;
                })
                .sorted((a, b) -> Integer.compare(b.getCount(), a.getCount()))
                .collect(Collectors.toList());
    }

    @Override
    public LibraryInspectionResponse inspectLibrary(LibraryInspectionRequest request) {
        LibraryInspectionResponse response = new LibraryInspectionResponse();

        int minChunkLength = request.getMinChunkLength() != null ? request.getMinChunkLength() : DEFAULT_MIN_CHUNK_LENGTH;
        int outdatedDays = request.getOutdatedDays() != null ? request.getOutdatedDays() : DEFAULT_OUTDATED_DAYS;
        int unaccessedDays = request.getUnaccessedDays() != null ? request.getUnaccessedDays() : DEFAULT_UNACCESSED_DAYS;

        List<KnowledgeDoc> allDocs = knowledgeDocMapper.selectList(null);
        List<KnowledgeChunk> allChunks = knowledgeChunkMapper.selectList(null);
        List<DocViewLog> allViewLogs = docViewLogMapper.selectList(null);
        List<KnowledgeCategory> categories = knowledgeCategoryMapper.selectList(null);
        Map<Long, String> categoryMap = categories.stream()
                .collect(Collectors.toMap(KnowledgeCategory::getId, KnowledgeCategory::getName));

        LibraryInspectionResponse.LibraryInspectionStats stats = new LibraryInspectionResponse.LibraryInspectionStats();
        stats.setTotalDocs(allDocs.size());
        stats.setTotalChunks(allChunks.size());
        stats.setLastInspectionTime(LocalDateTime.now());
        response.setStats(stats);

        List<LibraryInspectionResponse.DuplicateDocGroup> duplicateGroups = new ArrayList<>();
        if (request.getEnableDuplicateCheck() == null || request.getEnableDuplicateCheck()) {
            duplicateGroups = detectDuplicateDocs(allDocs, categoryMap);
            stats.setDuplicateDocGroups(duplicateGroups.size());
            stats.setDuplicateDocCount(duplicateGroups.stream().mapToInt(g -> g.getDocuments().size()).sum());
        } else {
            stats.setDuplicateDocGroups(0);
            stats.setDuplicateDocCount(0);
        }

        List<LibraryInspectionResponse.LowQualityChunk> lowQualityChunks = new ArrayList<>();
        if (request.getEnableQualityCheck() == null || request.getEnableQualityCheck()) {
            lowQualityChunks = detectLowQualityChunks(allChunks, allDocs, categoryMap, minChunkLength);
            stats.setLowQualityChunkCount(lowQualityChunks.size());
        } else {
            stats.setLowQualityChunkCount(0);
        }

        List<LibraryInspectionResponse.OutdatedDoc> outdatedDocs = new ArrayList<>();
        if (request.getEnableOutdatedCheck() == null || request.getEnableOutdatedCheck()) {
            outdatedDocs = detectOutdatedDocs(allDocs, categoryMap, outdatedDays);
            stats.setOutdatedDocCount(outdatedDocs.size());
        } else {
            stats.setOutdatedDocCount(0);
        }

        List<LibraryInspectionResponse.UnaccessedDoc> unaccessedDocs = new ArrayList<>();
        if (request.getEnableAccessCheck() == null || request.getEnableAccessCheck()) {
            unaccessedDocs = detectUnaccessedDocs(allDocs, allViewLogs, categoryMap, unaccessedDays);
            stats.setUnaccessedDocCount(unaccessedDocs.size());
        } else {
            stats.setUnaccessedDocCount(0);
        }

        response.setDuplicateDocs(duplicateGroups);
        response.setLowQualityChunks(lowQualityChunks);
        response.setOutdatedDocs(outdatedDocs);
        response.setUnaccessedDocs(unaccessedDocs);

        List<LibraryInspectionResponse.InspectionExport> exportData = new ArrayList<>();
        for (LibraryInspectionResponse.DuplicateDocGroup group : duplicateGroups) {
            for (LibraryInspectionResponse.DuplicateDoc doc : group.getDocuments()) {
                LibraryInspectionResponse.InspectionExport exp = new LibraryInspectionResponse.InspectionExport();
                exp.setType("重复文档");
                exp.setName(doc.getDocName());
                exp.setIssue("与" + (group.getDocuments().size() - 1) + "个文档重复");
                exp.setDetail("相似度: " + String.format("%.1f", group.getSimilarity() * 100) + "%");
                exportData.add(exp);
            }
        }
        for (LibraryInspectionResponse.LowQualityChunk chunk : lowQualityChunks) {
            LibraryInspectionResponse.InspectionExport exp = new LibraryInspectionResponse.InspectionExport();
            exp.setType("低质量Chunk");
            exp.setName(chunk.getDocName());
            exp.setIssue(chunk.getIssueType());
            exp.setDetail(chunk.getIssueDescription());
            exportData.add(exp);
        }
        for (LibraryInspectionResponse.OutdatedDoc doc : outdatedDocs) {
            LibraryInspectionResponse.InspectionExport exp = new LibraryInspectionResponse.InspectionExport();
            exp.setType("过期文档");
            exp.setName(doc.getDocName());
            exp.setIssue("超过" + doc.getDaySinceUpdate() + "天未更新");
            exp.setDetail("创建于: " + doc.getCreateTime());
            exportData.add(exp);
        }
        for (LibraryInspectionResponse.UnaccessedDoc doc : unaccessedDocs) {
            LibraryInspectionResponse.InspectionExport exp = new LibraryInspectionResponse.InspectionExport();
            exp.setType("无人访问文档");
            exp.setName(doc.getDocName());
            exp.setIssue("超过" + doc.getDaySinceAccess() + "天未被访问");
            exp.setDetail("访问次数: " + doc.getAccessCount());
            exportData.add(exp);
        }
        response.setExportData(exportData);

        return response;
    }

    @Override
    public Map<String, Object> getLibraryInspectionStats() {
        List<KnowledgeDoc> allDocs = knowledgeDocMapper.selectList(null);
        List<KnowledgeChunk> allChunks = knowledgeChunkMapper.selectList(null);
        List<DocViewLog> allViewLogs = docViewLogMapper.selectList(null);

        Map<String, Object> stats = new HashMap<>();
        stats.put("totalDocs", allDocs.size());
        stats.put("totalChunks", allChunks.size());
        stats.put("totalViewLogs", allViewLogs.size());

        LocalDateTime thirtyDaysAgo = LocalDateTime.now().minusDays(30);
        int recentAccessCount = (int) allViewLogs.stream()
                .filter(log -> log.getCreateTime().isAfter(thirtyDaysAgo))
                .count();
        stats.put("recentAccessCount", recentAccessCount);

        return stats;
    }

    @Override
    public UnansweredAnalysisResponse refreshUnanswered(UnansweredAnalysisRequest request) {
        return analyzeUnansweredQuestions(request);
    }

    @Override
    public LibraryInspectionResponse refreshLibrary(LibraryInspectionRequest request) {
        return inspectLibrary(request);
    }

    private List<LibraryInspectionResponse.DuplicateDocGroup> detectDuplicateDocs(
            List<KnowledgeDoc> docs, Map<Long, String> categoryMap) {
        List<LibraryInspectionResponse.DuplicateDocGroup> groups = new ArrayList<>();
        Set<Long> processedIds = new HashSet<>();
        int groupId = 1;

        for (KnowledgeDoc doc : docs) {
            if (processedIds.contains(doc.getId())) continue;
            if (!"COMPLETED".equals(doc.getStatus())) continue;

            List<KnowledgeDoc> similarDocs = new ArrayList<>();
            similarDocs.add(doc);
            processedIds.add(doc.getId());

            for (KnowledgeDoc other : docs) {
                if (processedIds.contains(other.getId())) continue;
                if (!"COMPLETED".equals(other.getStatus())) continue;

                // 文档名完全相同判定为重复（与 Python 侧一致）
                String name1 = doc.getDocName() != null ? doc.getDocName().trim() : "";
                String name2 = other.getDocName() != null ? other.getDocName().trim() : "";
                if (!name1.isEmpty() && name1.equals(name2)) {
                    similarDocs.add(other);
                    processedIds.add(other.getId());
                }
            }

            if (similarDocs.size() > 1) {
                LibraryInspectionResponse.DuplicateDocGroup group = new LibraryInspectionResponse.DuplicateDocGroup();
                group.setGroupId((long) groupId++);
                group.setSimilarity(1.0);
                group.setReason("文档名称相同");

                List<LibraryInspectionResponse.DuplicateDoc> groupDocs = similarDocs.stream()
                        .map(d -> {
                            LibraryInspectionResponse.DuplicateDoc dd = new LibraryInspectionResponse.DuplicateDoc();
                            dd.setId(d.getId());
                            dd.setDocName(d.getDocName());
                            dd.setCategoryId(d.getCategoryId());
                            dd.setCategoryName(categoryMap.getOrDefault(d.getCategoryId(), "未分类"));
                            dd.setCreateTime(d.getCreateTime());
                            return dd;
                        })
                        .collect(Collectors.toList());
                group.setDocuments(groupDocs);
                groups.add(group);
            }
        }

        return groups;
    }

    private List<LibraryInspectionResponse.LowQualityChunk> detectLowQualityChunks(
            List<KnowledgeChunk> chunks, List<KnowledgeDoc> docs,
            Map<Long, String> categoryMap, int minLength) {
        Map<Long, String> docNameMap = docs.stream()
                .collect(Collectors.toMap(KnowledgeDoc::getId, KnowledgeDoc::getDocName));

        return chunks.stream()
                .filter(chunk -> {
                    String text = chunk.getChunkText();
                    if (text == null || text.trim().isEmpty()) return true;
                    int length = text.trim().length();
                    if (length < minLength) return true;
                    if (length > DEFAULT_MAX_CHUNK_LENGTH) return true;
                    return false;
                })
                .map(chunk -> {
                    LibraryInspectionResponse.LowQualityChunk lqc = new LibraryInspectionResponse.LowQualityChunk();
                    lqc.setId(chunk.getId());
                    lqc.setDocId(chunk.getDocId());
                    lqc.setDocName(docNameMap.getOrDefault(chunk.getDocId(), "未知文档"));
                    lqc.setChunkText(chunk.getChunkText());
                    lqc.setChunkIndex(chunk.getChunkIndex());

                    String text = chunk.getChunkText();
                    if (text == null || text.trim().isEmpty()) {
                        lqc.setIssueType("空内容");
                        lqc.setIssueDescription("Chunk内容为空");
                    } else if (text.trim().length() < minLength) {
                        lqc.setIssueType("内容过短");
                        lqc.setIssueDescription("内容长度仅" + text.trim().length() + "字符，少于" + minLength + "字符");
                    } else {
                        lqc.setIssueType("内容过长");
                        lqc.setIssueDescription("内容长度" + text.trim().length() + "字符，超过" + DEFAULT_MAX_CHUNK_LENGTH + "字符");
                    }
                    return lqc;
                })
                .collect(Collectors.toList());
    }

    private List<LibraryInspectionResponse.OutdatedDoc> detectOutdatedDocs(
            List<KnowledgeDoc> docs, Map<Long, String> categoryMap, int days) {
        LocalDateTime threshold = LocalDateTime.now().minusDays(days);

        return docs.stream()
                .filter(doc -> "COMPLETED".equals(doc.getStatus()))
                .filter(doc -> doc.getCreateTime() != null && doc.getCreateTime().isBefore(threshold))
                .map(doc -> {
                    LibraryInspectionResponse.OutdatedDoc odd = new LibraryInspectionResponse.OutdatedDoc();
                    odd.setId(doc.getId());
                    odd.setDocName(doc.getDocName());
                    odd.setCategoryName(categoryMap.getOrDefault(doc.getCategoryId(), "未分类"));
                    odd.setCreateTime(doc.getCreateTime());
                    odd.setUpdateTime(doc.getCreateTime());
                    odd.setDaySinceUpdate((int) ChronoUnit.DAYS.between(doc.getCreateTime(), LocalDateTime.now()));
                    return odd;
                })
                .sorted(Comparator.comparingInt(LibraryInspectionResponse.OutdatedDoc::getDaySinceUpdate).reversed())
                .collect(Collectors.toList());
    }

    private List<LibraryInspectionResponse.UnaccessedDoc> detectUnaccessedDocs(
            List<KnowledgeDoc> docs, List<DocViewLog> viewLogs,
            Map<Long, String> categoryMap, int days) {
        LocalDateTime threshold = LocalDateTime.now().minusDays(days);

        Map<Long, List<DocViewLog>> docViewsMap = viewLogs.stream()
                .collect(Collectors.groupingBy(DocViewLog::getDocId));

        Map<Long, LocalDateTime> lastAccessMap = new HashMap<>();
        Map<Long, Integer> accessCountMap = new HashMap<>();
        for (DocViewLog log : viewLogs) {
            lastAccessMap.merge(log.getDocId(), log.getCreateTime(), (a, b) -> a.isAfter(b) ? a : b);
            accessCountMap.merge(log.getDocId(), 1, Integer::sum);
        }

        return docs.stream()
                .filter(doc -> "COMPLETED".equals(doc.getStatus()))
                .filter(doc -> {
                    LocalDateTime lastAccess = lastAccessMap.get(doc.getId());
                    return lastAccess == null || lastAccess.isBefore(threshold);
                })
                .map(doc -> {
                    LibraryInspectionResponse.UnaccessedDoc uad = new LibraryInspectionResponse.UnaccessedDoc();
                    uad.setId(doc.getId());
                    uad.setDocName(doc.getDocName());
                    uad.setCategoryName(categoryMap.getOrDefault(doc.getCategoryId(), "未分类"));
                    uad.setCreateTime(doc.getCreateTime());
                    uad.setLastAccessTime(lastAccessMap.get(doc.getId()));
                    uad.setAccessCount(accessCountMap.getOrDefault(doc.getId(), 0));

                    LocalDateTime lastAccess = lastAccessMap.get(doc.getId());
                    if (lastAccess != null) {
                        uad.setDaySinceAccess((int) ChronoUnit.DAYS.between(lastAccess, LocalDateTime.now()));
                    } else {
                        uad.setDaySinceAccess((int) ChronoUnit.DAYS.between(doc.getCreateTime(), LocalDateTime.now()));
                    }
                    return uad;
                })
                .sorted(Comparator.comparingInt(LibraryInspectionResponse.UnaccessedDoc::getDaySinceAccess).reversed())
                .collect(Collectors.toList());
    }
}