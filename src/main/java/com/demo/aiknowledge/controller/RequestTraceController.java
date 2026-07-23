package com.demo.aiknowledge.controller;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.demo.aiknowledge.entity.RequestTrace;
import com.demo.aiknowledge.mapper.RequestTraceMapper;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 全链路追踪查询 API（供管理端查看）。
 * /summaries     — 请求列表（不含 trace_json，避免数据量过大）
 * /{traceId}     — 单条 trace 的 events 数组
 */
@Slf4j
@RestController
@RequestMapping("/api/v1/traces")
@RequiredArgsConstructor
public class RequestTraceController {

    private final RequestTraceMapper requestTraceMapper;
    private final ObjectMapper objectMapper;

    /** 请求列表（摘要，不含 trace_json） */
    @GetMapping("/summaries")
    public List<Map<String, Object>> getSummaries(@RequestParam(defaultValue = "50") int limit) {
        return requestTraceMapper.selectSummaries(limit);
    }

    /** 按 traceId 查询完整 trace_json（含 events、durationMs、status 等） */
    @GetMapping("/{traceId}")
    public Object getByTraceId(@PathVariable String traceId) {
        RequestTrace row = requestTraceMapper.selectByTraceId(traceId);
        if (row == null || row.getTraceJson() == null) {
            return java.util.Map.of("events", List.of());
        }
        try {
            return objectMapper.readValue(row.getTraceJson(), Map.class);
        } catch (Exception e) {
            log.warn("Failed to parse trace_json for traceId={}", traceId, e);
            return java.util.Map.of("events", List.of());
        }
    }

    /** 分页查询 + 条件过滤，每条记录附带从 trace_json 提取的 startTime */
    @GetMapping("/list")
    public Map<String, Object> list(
            @RequestParam(defaultValue = "1") int pageNum,
            @RequestParam(defaultValue = "10") int pageSize,
            @RequestParam(required = false) Long userId,
            @RequestParam(required = false) String userType,
            @RequestParam(required = false) String status,
            @RequestParam(required = false) String startTime,
            @RequestParam(required = false) String endTime) {

        LambdaQueryWrapper<RequestTrace> wrapper = new LambdaQueryWrapper<>();
        if (userId != null) {
            wrapper.eq(RequestTrace::getUserId, userId);
        }
        if ("user".equals(userType)) {
            wrapper.notLike(RequestTrace::getSessionId, "admin\\_%");
        } else if ("admin".equals(userType)) {
            wrapper.like(RequestTrace::getSessionId, "admin\\_");
        }
        if (StringUtils.hasText(status)) {
            wrapper.eq(RequestTrace::getStatus, status);
        }
        if (StringUtils.hasText(startTime)) {
            wrapper.ge(RequestTrace::getCreatedAt, startTime + "T00:00:00");
        }
        if (StringUtils.hasText(endTime)) {
            wrapper.le(RequestTrace::getCreatedAt, endTime + "T23:59:59");
        }
        wrapper.orderByDesc(RequestTrace::getCreatedAt);

        Page<RequestTrace> page = requestTraceMapper.selectPage(
                new Page<>(pageNum, pageSize), wrapper);

        // 从 trace_json 中提取 startTime，并清掉大体积的 trace_json
        for (RequestTrace row : page.getRecords()) {
            if (row.getTraceJson() != null) {
                try {
                    Map<String, Object> json = objectMapper.readValue(row.getTraceJson(), Map.class);
                    row.setStartTime((String) json.getOrDefault("startTime", null));
                } catch (Exception ignored) {}
            }
            row.setTraceJson(null);
        }

        Map<String, Object> result = new HashMap<>();
        result.put("records", page.getRecords());
        result.put("total", page.getTotal());
        return result;
    }

    /** 按 traceId 删除 */
    @DeleteMapping("/{traceId}")
    public String delete(@PathVariable String traceId) {
        requestTraceMapper.delete(new LambdaQueryWrapper<RequestTrace>()
                .eq(RequestTrace::getTraceId, traceId));
        return "ok";
    }
}
