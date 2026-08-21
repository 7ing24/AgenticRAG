package com.demo.aiknowledge.scheduler;

import com.demo.aiknowledge.dto.LibraryInspectionRequest;
import com.demo.aiknowledge.dto.UnansweredAnalysisRequest;
import com.demo.aiknowledge.service.KnowledgeInspectionService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

/**
 * 知识巡检后台定时任务
 *
 * 用固定默认参数预计算未命中分析 + 知识库巡检，结果缓存在 Service 中。
 * 前端默认请求直接命中缓存即时展示，无需等待耗时的语义聚类（Python 调用）。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class KnowledgeInspectionScheduler {

    private final KnowledgeInspectionService knowledgeInspectionService;

    @Scheduled(
            fixedDelayString = "${ai.inspection.interval-ms:1800000}",
            initialDelayString = "${ai.inspection.initial-delay-ms:10000}")
    public void runInspections() {
        log.info("[InspectionScheduler] 定时巡检开始...");
        try {
            // 默认参数：不限定日期，minCount=1，clusterThreshold=0.85（与前端默认请求一致，保证命中缓存）
            knowledgeInspectionService.refreshUnanswered(new UnansweredAnalysisRequest());
        } catch (Exception e) {
            log.error("[InspectionScheduler] 未命中问题分析失败: {}", e.getMessage(), e);
        }
        try {
            knowledgeInspectionService.refreshLibrary(new LibraryInspectionRequest());
        } catch (Exception e) {
            log.error("[InspectionScheduler] 知识库巡检失败: {}", e.getMessage(), e);
        }
        log.info("[InspectionScheduler] 定时巡检完成");
    }
}
