package com.demo.aiknowledge.controller;

import com.demo.aiknowledge.common.Result;
import com.demo.aiknowledge.entity.KnowledgeDoc;
import com.demo.aiknowledge.service.KnowledgeService;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.List;

@RestController
@RequestMapping("/api/knowledge")
@RequiredArgsConstructor
public class KnowledgeController {

    private final KnowledgeService knowledgeService;

    @PostMapping("/upload")
    public Result<KnowledgeDoc> upload(@RequestParam("file") MultipartFile file, @RequestParam(required = false) Long categoryId) {
        return Result.success(knowledgeService.uploadDoc(file, categoryId));
    }

    @GetMapping("/list")
    public Result<List<KnowledgeDoc>> list(@RequestParam(required = false) Long categoryId) {
        return Result.success(knowledgeService.listDocs(categoryId));
    }

    @DeleteMapping("/{id}")
    public Result<String> delete(@PathVariable Long id) {
        knowledgeService.deleteDoc(id);
        return Result.success("Deleted successfully");
    }

    @GetMapping("/view/{id}")
    public Result<KnowledgeDoc> view(@PathVariable Long id, @RequestParam Long userId) {
        return Result.success(knowledgeService.viewDoc(id, userId));
    }

    @GetMapping("/view-file/{id}")
    public void viewFile(@PathVariable Long id, HttpServletResponse response) throws IOException {
        KnowledgeDoc doc = knowledgeService.viewDoc(id, null);
        if (doc == null || doc.getFilePath() == null) {
            response.sendError(404, "文件不存在");
            return;
        }

        File file = new File(doc.getFilePath());
        if (!file.exists()) {
            response.sendError(404, "文件不存在");
            return;
        }

        String fileName = doc.getDocName() != null ? doc.getDocName() : file.getName();
        String ext = fileName.contains(".") ? fileName.substring(fileName.lastIndexOf(".") + 1).toLowerCase() : "";

        String contentType = switch (ext) {
            case "pdf" -> "application/pdf";
            case "txt", "md" -> "text/plain; charset=UTF-8";
            case "doc", "docx" -> "application/msword";
            default -> "application/octet-stream";
        };

        response.setContentType(contentType);
        response.setHeader("Content-Disposition", "inline; filename=\"" +
                URLEncoder.encode(fileName, StandardCharsets.UTF_8) + "\"");

        try (InputStream in = new FileInputStream(file);
             OutputStream out = response.getOutputStream()) {
            byte[] buffer = new byte[8192];
            int len;
            while ((len = in.read(buffer)) != -1) {
                out.write(buffer, 0, len);
            }
        }
    }
}
