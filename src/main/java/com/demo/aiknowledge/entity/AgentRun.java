package com.demo.aiknowledge.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

@Data
@TableName("agent_run")
public class AgentRun {
    @TableId(type = IdType.ASSIGN_UUID)
    private String id;

    private String runId;
    private String traceId;
    private String parentRunId;
    private String agentType;
    private String conversationId;
    private String userId;
    private String status;
    private String goal;
    private LocalDateTime startTime;
    private LocalDateTime endTime;
    private String input;
    private String output;
    private String errorMessage;
    private String errorCode;
    private LocalDateTime createdAt;

    /** 同 trace 下的子 Agent 数量（非 DB 字段） */
    @TableField(exist = false)
    private Integer childCount;

    /** 步骤记录（非 DB 字段，详情查询时填充） */
    @TableField(exist = false)
    private List<AgentStep> steps;
}
