package com.demo.aiknowledge.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.DispatcherType;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.MediaType;
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

import lombok.RequiredArgsConstructor;

import java.util.Arrays;
import java.util.Map;

@Configuration
@EnableWebSecurity
@EnableMethodSecurity
@RequiredArgsConstructor
public class SecurityConfig {

    private final JwtFilter jwtFilter;

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
            // 禁用CSRF保护，对于前后端分离的应用
            .csrf(csrf -> csrf.disable())
            // 启用CORS支持
            .cors(cors -> cors.configurationSource(corsConfigurationSource()))
            // 配置请求授权
            .authorizeHttpRequests(auth -> auth
                // 允许异步转发（SSE/Streaming 需要）
                .dispatcherTypeMatchers(DispatcherType.ASYNC).permitAll()
                // 允许OPTIONS预检请求
                .requestMatchers(org.springframework.http.HttpMethod.OPTIONS, "/**").permitAll()
                // 允许所有/api/auth下的请求
                .requestMatchers("/api/auth/**").permitAll()
                // 允许所有/api/admin/login请求
                .requestMatchers("/api/admin/login").permitAll()
                // 允许图片访问路径
                .requestMatchers("/api/chat/view/image/**").permitAll()
                // 允许文档文件预览（新标签页打开，不带 token）
                .requestMatchers("/api/knowledge/view-file/**").permitAll()
                // 允许缓存监控端点（内部指标采集）
                .requestMatchers("/api/cache/**").permitAll()
                // 管理员接口需要ADMIN角色
                .requestMatchers("/api/admin/**").hasRole("ADMIN")
                // 管理端对话接口
                .requestMatchers("/api/admin-chat/**").hasAnyRole("USER", "ADMIN")
                // 用户接口需要USER或ADMIN角色
                .requestMatchers("/api/chat/**").hasAnyRole("USER", "ADMIN")
                .requestMatchers("/api/knowledge/**").hasAnyRole("USER", "ADMIN")
                // 其他请求需要认证
                .anyRequest().authenticated()
            )
            // 添加JWT过滤器
            .addFilterBefore(jwtFilter, UsernamePasswordAuthenticationFilter.class)
            // 禁用默认的登录表单
            .formLogin(form -> form.disable())
            // 禁用默认的HTTP基本认证
            .httpBasic(httpBasic -> httpBasic.disable());

        // 异常处理：未认证返回401，权限不足返回403
        http.exceptionHandling(ex -> ex
            .authenticationEntryPoint((request, response, authException) -> {
                response.setContentType(MediaType.APPLICATION_JSON_VALUE);
                response.setCharacterEncoding("UTF-8");
                response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
                new ObjectMapper().writeValue(response.getWriter(),
                    Map.of("code", 401, "message", "未登录或登录已过期"));
            })
            .accessDeniedHandler((request, response, accessDeniedException) -> {
                response.setContentType(MediaType.APPLICATION_JSON_VALUE);
                response.setCharacterEncoding("UTF-8");
                response.setStatus(HttpServletResponse.SC_FORBIDDEN);
                new ObjectMapper().writeValue(response.getWriter(),
                    Map.of("code", 403, "message", "权限不足"));
            })
        );

        // 配置无状态会话管理
        http.sessionManagement(session -> session
            .sessionCreationPolicy(SessionCreationPolicy.STATELESS)
        );

        return http.build();
    }

    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration configuration = new CorsConfiguration();
        configuration.setAllowedOrigins(Arrays.asList("http://localhost:3000")); // 前端开发服务器
        configuration.setAllowedMethods(Arrays.asList("GET", "POST", "PUT", "DELETE", "OPTIONS"));
        configuration.setAllowedHeaders(Arrays.asList("*"));
        configuration.setAllowCredentials(true);
        configuration.setExposedHeaders(Arrays.asList("Authorization")); // 允许前端访问Authorization header

        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/api/**", configuration);
        return source;
    }
}
