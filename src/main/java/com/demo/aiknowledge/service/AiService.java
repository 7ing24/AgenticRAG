package com.demo.aiknowledge.service;

import com.demo.aiknowledge.dto.AiResponse;

public interface AiService {
    void parseDocument(String filePath, Long docId);

    /**
     * 回答问题（上下文由 Python MemoryAgent 统一管理）
     * @param question 用户问题
     * @param userId 用户ID
     * @param conversationId 会话ID
     * @return AI回答对象
     */
    AiResponse ask(String question, Long userId, Long conversationId);

    /**
     * 回答问题（带全链路追踪）
     * @param question 用户问题
     * @param userId 用户ID
     * @param conversationId 会话ID
     * @param traceId 全链路追踪ID
     * @return AI回答对象
     */
    AiResponse ask(String question, Long userId, Long conversationId, String traceId);

    /**
     * 生成会话标题并更新数据库
     * @param conversationId 会话ID
     * @param question 用户问题
     */
    void generateTitle(Long conversationId, String question);

    /**
     * 删除文档向量索引
     */
    void deleteDoc(Long docId);

    /**
     * 管理端AI助手问答（上下文由 Python MemoryAgent 统一管理）
     * @param question 用户问题
     * @param adminId 管理员ID
     * @param conversationId 会话ID
     * @param traceId 全链路追踪ID
     * @return AI回答对象（含 traces 字段）
     */
    java.util.Map<String, Object> askForAdmin(String question, Long adminId, Long conversationId, String traceId);

    /**
     * 语义缓存查找：查找与问题语义相似的已缓存问题
     * @param question 用户问题
     * @return 匹配的缓存 key，未找到返回 null
     */
    String semanticCacheLookup(String question);

    /**
     * 向语义缓存索引中添加问题（异步，不阻塞）
     * @param question 已缓存的问题文本
     */
    void addToSemanticCache(String question);
}
