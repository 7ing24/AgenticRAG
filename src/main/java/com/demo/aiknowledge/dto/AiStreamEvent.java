package com.demo.aiknowledge.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.NoArgsConstructor;

/** SSE 流式事件 — Python /ask/stream 的单个 token/end/error 事件 */
@Data
@NoArgsConstructor
public class AiStreamEvent {
    private String type;

    /** token: 当前文本片段; end: 完整回答 */
    @JsonProperty("content")
    private String content;

    /** end 事件的完整回答（Python 端 content 字段即为完整回答） */
    public String getAnswer() { return content; }

    /** 任务类型（knowledge_qa / chitchat 等） */
    @JsonProperty("task_type")
    private String taskType;
}
