package com.demo.aiknowledge.controller;

import com.demo.aiknowledge.common.Result;
import com.demo.aiknowledge.dto.FeedbackRequest;
import com.demo.aiknowledge.entity.AdminConversation;
import com.demo.aiknowledge.entity.AdminMessage;
import com.demo.aiknowledge.mapper.AdminMessageMapper;
import com.demo.aiknowledge.service.AdminChatService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/api/admin-chat")
@RequiredArgsConstructor
@Slf4j
public class AdminChatController {

    private final AdminChatService adminChatService;
    private final WebClient webClient;
    private final AdminMessageMapper adminMessageMapper;

    @PostMapping("/conversations")
    public Result<AdminConversation> createConversation(
            @RequestParam Long adminId,
            @RequestParam(required = false) String title) {
        return Result.success(adminChatService.createConversation(adminId, title));
    }

    @GetMapping("/conversations")
    public Result<List<AdminConversation>> getHistory(@RequestParam Long adminId) {
        return Result.success(adminChatService.getHistory(adminId));
    }

    @PostMapping("/messages")
    public Result<AdminMessage> sendMessage(
            @RequestParam Long adminId,
            @RequestParam Long conversationId,
            @RequestBody Map<String, String> request) {
        String content = request.get("content");
        return Result.success(adminChatService.sendMessage(adminId, conversationId, content));
    }

    @PostMapping(value = "/stream/messages", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public ResponseEntity<SseEmitter> sendMessageStream(
            @RequestParam Long adminId,
            @RequestParam Long conversationId,
            @RequestBody Map<String, String> request) {
        String content = request.get("content");
        String traceId = UUID.randomUUID().toString();

        // Save user message
        AdminMessage userMsg = new AdminMessage();
        userMsg.setConversationId(conversationId);
        userMsg.setRole("user");
        userMsg.setContent(content);
        userMsg.setCreateTime(LocalDateTime.now());
        adminMessageMapper.insert(userMsg);

        // Build request body for Python
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("question", content);
        requestBody.put("is_admin", true);
        requestBody.put("conversation_id", conversationId.toString());
        requestBody.put("user_id", adminId.toString());
        requestBody.put("username", "admin_" + adminId);
        requestBody.put("trace_id", traceId);

        log.info("Admin streaming request: adminId={}, conversationId={}, traceId={}", adminId, conversationId, traceId);

        SseEmitter emitter = new SseEmitter(180000L);

        webClient.post()
                .uri("/api/ask/stream")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(requestBody)
                .accept(MediaType.TEXT_EVENT_STREAM)
                .retrieve()
                .bodyToFlux(String.class)
                .subscribe(
                    line -> {
                        if (line.isEmpty()) return;
                        try {
                            emitter.send(SseEmitter.event().data(line.replaceFirst("^data: ", "")));
                        } catch (IOException e) {
                            emitter.completeWithError(e);
                        }
                    },
                    error -> {
                        log.error("Admin stream error for traceId={}: {}", traceId, error.getMessage());
                        emitter.completeWithError(error);
                    },
                    () -> emitter.complete()
                );

        return ResponseEntity.ok()
                .header(HttpHeaders.CACHE_CONTROL, "no-cache")
                .header("X-Accel-Buffering", "no")
                .body(emitter);
    }

    @GetMapping("/messages")
    public Result<List<AdminMessage>> getMessages(@RequestParam Long conversationId) {
        return Result.success(adminChatService.getMessages(conversationId));
    }

    @DeleteMapping("/conversations/{id}")
    public Result<String> deleteConversation(@PathVariable Long id) {
        adminChatService.deleteConversation(id);
        return Result.success("Conversation deleted");
    }

    @PutMapping("/conversations/{id}")
    public Result<AdminConversation> updateConversation(
            @PathVariable Long id,
            @RequestBody AdminConversation conversation) {
        return Result.success(adminChatService.updateConversation(id, conversation.getTitle(), conversation.getIsPinned()));
    }

    @PostMapping("/messages/feedback")
    public Result<AdminMessage> submitFeedback(@RequestBody FeedbackRequest request) {
        return Result.success(adminChatService.submitFeedback(request.getMessageId(), request.getFeedbackType()));
    }
}