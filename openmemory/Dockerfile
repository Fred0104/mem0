# 使用已构建的 API 和 UI 镜像
FROM mem0ai/openmemory-mcp:latest AS backend
FROM mem0ai/openmemory-ui:latest AS frontend

# 使用 Nginx 作为统一入口
FROM nginx:alpine
COPY --from=backend /usr/src/openmemory /usr/share/nginx/html/api
COPY --from=frontend /app /usr/share/nginx/html/ui
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
