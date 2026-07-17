package com.demo.aiknowledge.controller;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.demo.aiknowledge.common.Result;
import com.demo.aiknowledge.entity.AgentRun;
import com.demo.aiknowledge.entity.AgentStep;
import com.demo.aiknowledge.mapper.AgentRunMapper;
import com.demo.aiknowledge.mapper.AgentStepMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.*;

@RestController
@RequestMapping("/api/agent-run")
@RequiredArgsConstructor
@Slf4j
public class AgentRunController {

    private final AgentRunMapper agentRunMapper;
    private final AgentStepMapper agentStepMapper;

    /**
     * 列表查询 — 默认只返回顶层 run（parent_run_id IS NULL），即 trace 级别视图
     */
    @GetMapping("/list")
    public Result<IPage<AgentRun>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String userId,
            @RequestParam(required = false) String status,
            @RequestParam(required = false) LocalDateTime startTime,
            @RequestParam(required = false) LocalDateTime endTime,
            @RequestParam(required = false) String parentRunId,
            @RequestParam(required = false) String traceId) {

        Page<AgentRun> page = new Page<>(pageNum, pageSize);
        LambdaQueryWrapper<AgentRun> wrapper = new LambdaQueryWrapper<>();

        if (userId != null && !userId.isEmpty()) {
            wrapper.eq(AgentRun::getUserId, userId);
        }
        if (status != null && !status.isEmpty()) {
            wrapper.eq(AgentRun::getStatus, status);
        }
        if (startTime != null) {
            wrapper.ge(AgentRun::getStartTime, startTime);
        }
        if (endTime != null) {
            wrapper.le(AgentRun::getStartTime, endTime);
        }
        if (traceId != null && !traceId.isEmpty()) {
            wrapper.eq(AgentRun::getTraceId, traceId);
        }
        if (parentRunId != null) {
            if (parentRunId.equals("null") || parentRunId.isEmpty()) {
                wrapper.isNull(AgentRun::getParentRunId);
            } else {
                wrapper.eq(AgentRun::getParentRunId, parentRunId);
            }
        } else {
            // 默认只显示顶层 run（trace 级别）
            wrapper.isNull(AgentRun::getParentRunId);
        }

        wrapper.orderByDesc(AgentRun::getStartTime);
        IPage<AgentRun> result = agentRunMapper.selectPage(page, wrapper);

        // 填充子 Agent 数量
        if (!result.getRecords().isEmpty()) {
            Set<String> parentRunIds = new HashSet<>();
            for (AgentRun r : result.getRecords()) {
                parentRunIds.add(r.getRunId());
            }
            // 统计每个顶层 run 的子 run 数量
            LambdaQueryWrapper<AgentRun> countWrapper = new LambdaQueryWrapper<>();
            countWrapper.in(AgentRun::getParentRunId, parentRunIds);
            List<AgentRun> children = agentRunMapper.selectList(countWrapper);
            Map<String, Integer> childCounts = new HashMap<>();
            for (AgentRun child : children) {
                String pid = child.getParentRunId();
                childCounts.put(pid, childCounts.getOrDefault(pid, 0) + 1);
            }
            for (AgentRun r : result.getRecords()) {
                r.setChildCount(childCounts.getOrDefault(r.getRunId(), 0));
            }
        }

        return Result.success(result);
    }

    /**
     * 获取一条 trace 的完整链路（所有 run + steps）
     */
    @GetMapping("/trace/{traceId}")
    public Result<List<AgentRun>> getTrace(@PathVariable String traceId) {
        LambdaQueryWrapper<AgentRun> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(AgentRun::getTraceId, traceId)
               .orderByAsc(AgentRun::getStartTime);
        List<AgentRun> runs = agentRunMapper.selectList(wrapper);

        // 填充每个 run 的 steps
        for (AgentRun run : runs) {
            List<AgentStep> steps = agentStepMapper.selectList(
                    new LambdaQueryWrapper<AgentStep>()
                            .eq(AgentStep::getRunId, run.getRunId())
                            .orderByAsc(AgentStep::getCreatedAt));
            run.setSteps(steps);
        }

        return Result.success(runs);
    }

    @GetMapping("/{runId}")
    public Result<AgentRun> getById(@PathVariable String runId) {
        AgentRun agentRun = agentRunMapper.selectOne(
                new LambdaQueryWrapper<AgentRun>().eq(AgentRun::getRunId, runId));
        if (agentRun == null) {
            return Result.error("AgentRun不存在");
        }
        // 填充 steps
        List<AgentStep> steps = agentStepMapper.selectList(
                new LambdaQueryWrapper<AgentStep>()
                        .eq(AgentStep::getRunId, runId)
                        .orderByAsc(AgentStep::getCreatedAt));
        agentRun.setSteps(steps);
        return Result.success(agentRun);
    }

    @GetMapping("/{runId}/steps")
    public Result<List<AgentStep>> getSteps(@PathVariable String runId) {
        List<AgentStep> steps = agentStepMapper.selectList(
                new LambdaQueryWrapper<AgentStep>()
                        .eq(AgentStep::getRunId, runId)
                        .orderByAsc(AgentStep::getCreatedAt));
        return Result.success(steps);
    }

    @GetMapping("/{runId}/tool-calls")
    public Result<List<?>> getToolCalls(@PathVariable String runId) {
        return Result.success(List.of());
    }

    @DeleteMapping("/{runId}")
    public Result<Void> delete(@PathVariable String runId) {
        int deleted = agentRunMapper.delete(
                new LambdaQueryWrapper<AgentRun>().eq(AgentRun::getRunId, runId));
        if (deleted > 0) {
            return Result.success(null);
        }
        return Result.error("删除失败");
    }
}
