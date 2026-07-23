package com.demo.aiknowledge.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.demo.aiknowledge.entity.RequestTrace;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import java.util.List;
import java.util.Map;

@Mapper
public interface RequestTraceMapper extends BaseMapper<RequestTrace> {

    @Select("SELECT * FROM request_trace WHERE trace_id = #{traceId}")
    RequestTrace selectByTraceId(String traceId);

    @Select("SELECT * FROM request_trace WHERE session_id = #{sessionId} ORDER BY created_at DESC")
    List<RequestTrace> selectBySessionId(String sessionId);

    @Select("SELECT id, trace_id AS traceId, session_id AS sessionId, user_id AS userId,"
            + " status, event_count AS eventCount, duration_ms AS durationMs,"
            + " JSON_UNQUOTE(JSON_EXTRACT(trace_json, '$.startTime')) AS startTime,"
            + " created_at AS createdAt"
            + " FROM request_trace ORDER BY created_at DESC LIMIT #{limit}")
    List<Map<String, Object>> selectSummaries(int limit);
}
