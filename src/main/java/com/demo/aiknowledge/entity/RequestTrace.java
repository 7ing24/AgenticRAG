package com.demo.aiknowledge.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 全链路请求追踪实体，对应 request_trace 表。
 * 一个 trace 占一行，事件列表序列化在 trace_json 列中。
 */
@Data
@TableName("request_trace")
public class RequestTrace {
    @TableId(type = IdType.AUTO)
    private Long id;
    /** 请求唯一标识 */
    private String traceId;
    /** 会话ID */
    private String sessionId;
    /** 用户ID */
    private Long userId;
    /** SUCCESS / FAILED */
    private String status;
    /** 事件总数 */
    private Integer eventCount;
    /** 总耗时(ms) */
    private Long durationMs;
    /** 失败时的错误摘要 */
    private String errorMessage;
    /** 完整事件列表 JSON */
    private String traceJson;
    /** 创建时间 */
    private LocalDateTime createdAt;

    // ── 非 DB 字段 ──

    /** 请求开始时间，从 trace_json.startTime 提取（非DB字段） */
    @TableField(exist = false)
    private String startTime;

    /** trace_json 解析后的事件列表（非DB字段） */
    @TableField(exist = false)
    private List<Map<String, Object>> events;
}
