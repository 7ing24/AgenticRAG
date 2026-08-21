package com.demo.aiknowledge.service;

import com.demo.aiknowledge.dto.LibraryInspectionRequest;
import com.demo.aiknowledge.dto.LibraryInspectionResponse;
import com.demo.aiknowledge.dto.UnansweredAnalysisRequest;
import com.demo.aiknowledge.dto.UnansweredAnalysisResponse;

import java.util.Map;

public interface KnowledgeInspectionService {
    UnansweredAnalysisResponse analyzeUnansweredQuestions(UnansweredAnalysisRequest request);
    Map<String, Object> getUnansweredStatistics();
    LibraryInspectionResponse inspectLibrary(LibraryInspectionRequest request);
    Map<String, Object> getLibraryInspectionStats();
    /** 强制重新计算未命中分析并更新缓存（定时任务/手动触发用） */
    UnansweredAnalysisResponse refreshUnanswered(UnansweredAnalysisRequest request);
    /** 强制重新计算知识库巡检并更新缓存（定时任务用） */
    LibraryInspectionResponse refreshLibrary(LibraryInspectionRequest request);
}