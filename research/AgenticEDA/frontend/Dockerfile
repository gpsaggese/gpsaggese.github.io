FROM node:22-alpine AS builder

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci

COPY . .

# Left empty so the client falls back to inferApiBase() in src/lib/backend.ts,
# which derives the API origin from window.location at runtime. Set this to a
# fixed URL only when the backend is not reachable from the browser's origin.
ENV NEXT_PUBLIC_API_URL=""
RUN npm run build

# --- Production ---
FROM node:22-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV HOSTNAME=0.0.0.0
ENV PORT=3000

COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

EXPOSE 3000
CMD ["node", "server.js"]
