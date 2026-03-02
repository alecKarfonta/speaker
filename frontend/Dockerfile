# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Copy package files
COPY package.json ./
COPY tsconfig.json ./
COPY tsconfig.node.json ./
COPY postcss.config.js ./
COPY tailwind.config.js ./

# Install dependencies
RUN npm install

# Copy source files
COPY . .

# Build the app
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built assets from build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx template (envsubst replaces ${TTS_BACKEND_HOST} at startup)
COPY nginx.conf.template /etc/nginx/templates/default.conf.template

# Default backend (can be overridden via docker-compose environment)
ENV TTS_BACKEND_HOST=moss-tts:8000

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]