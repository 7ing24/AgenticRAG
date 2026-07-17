package com.demo.aiknowledge.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import java.util.List;
import java.util.Map;

@Data
public class AiResponse {
    private String answer;
    private List<Map<String, Object>> sources;
    private String taskType;
    private List<Map<String, Object>> steps;
    private String traceId;
    private List<AgentRunRecord> runs;

    @Data
    public static class AgentRunRecord {
        @JsonProperty("run_id")
        private String runId;
        @JsonProperty("parent_run_id")
        private String parentRunId;
        @JsonProperty("agent_type")
        private String agentType;
        private String question;
        private List<Map<String, Object>> steps;
    }
}
