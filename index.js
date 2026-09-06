import { Container } from "@cloudflare/containers";

/**
 * Proxies all HTTP traffic to the FastAPI container (uvicorn on :8080).
 * Secrets TELEGRAM_TOKEN (+ optional WEBHOOK_SECRET) must be set on the Worker
 * and are forwarded into the container as env vars.
 */
export class BotContainer extends Container {
  defaultPort = 8080;
  // Keep warm during festival days; cold starts break Telegram webhooks.
  sleepAfter = "2h";
  enableInternet = true;

  constructor(ctx, env) {
    super(ctx, env);
    this.envVars = {
      TELEGRAM_TOKEN: env.TELEGRAM_TOKEN || "",
      WEBHOOK_SECRET: env.WEBHOOK_SECRET || "",
    };
  }
}

export default {
  async fetch(request, env) {
    const container = env.BOT_CONTAINER.getByName("fastapi");
    return container.fetch(request);
  },
};
