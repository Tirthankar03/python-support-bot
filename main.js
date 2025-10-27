import { Hono } from "hono";
import { GoogleGenerativeAI } from "@google/generative-ai";
import Postgres from "postgres";
import Redis from "ioredis";

// For WhatsApp (placeholder, swap with Telegram for now)
import TelegramBot from "node-telegram-bot-api"; // Replace with whatsapp-web.js or Twilio later

const app = new Hono();
const genAI = new GoogleGenerativeAI(process.env["GEMINI_API_KEY"]);
const bot = new TelegramBot(process.env["TELEGRAM_BOT_TOKEN"], { polling: true }); // Swap for WhatsApp client
const sql = Postgres(process.env["DATABASE_URL"]);
const redis = new Redis(process.env["REDIS_URL"]);

// Tools
async function get_current_date() {
  const today = new Date();
  return {
    success: true,
    result: {
      day: today.getDate(),
      month: today.getMonth() + 1,
      year: today.getFullYear(),
      formatted: today.toLocaleDateString("en-GB", { day: "2-digit", month: "2-digit", year: "numeric" }),
    },
  };
}

async function create_ticket({ chatId, problem, solution }) {
  await sql`INSERT INTO users (user_id, name) VALUES (${chatId}, ${name}) ON CONFLICT (user_id) DO UPDATE SET name = ${name}`;
  const [ticket] = await sql`
    INSERT INTO tickets (user_id, problem_description, solution_suggested)
    VALUES (${chatId}, ${problem}, ${solution})
    RETURNING ticket_id
  `;
  return { success: true, ticketId: ticket.ticket_id, message: `Ticket created: TICKET-${ticket.ticket_id.toString().padStart(4, "0")}` };
}

async function get_past_tickets({ chatId }) {
  const tickets = await sql`SELECT * FROM tickets WHERE user_id = ${chatId} ORDER BY created_at DESC LIMIT 5`;
  return { success: true, tickets };
}

async function save_message(chatId, message, isBot) {
  const msg = { message, isBot, timestamp: Date.now() };
  await redis.lpush(`user:${chatId}:history`, JSON.stringify(msg));
  await redis.ltrim(`user:${chatId}:history`, 0, 99); // Keep last 100 messages for long-term memory
  // No TTL for long-term storage, or set to 1 year: await redis.expire(`user:${chatId}:history`, 365 * 24 * 60 * 60);
}

async function get_conversation_history(chatId) {
  const messages = await redis.lrange(`user:${chatId}:history`, 0, 9); // Get last 10 for AI context
  return messages.map(msg => JSON.parse(msg)).reverse(); // Reverse to chronological order
}

const functions = {
  get_current_date,
  create_ticket,
  get_past_tickets,
};

const tools = [
  {
    functionDeclarations: [
      {
        name: "create_ticket",
        description: "Creates a new support ticket.",
        parameters: {
          type: "object",
          properties: {
            chatId: { type: "string" },
            problem: { type: "string" },
            solution: { type: "string" },
          },
          required: ["chatId", "problem", "solution"],
        },
      },
      {
        name: "get_past_tickets",
        description: "Fetches past tickets for a user.",
        parameters: {
          type: "object",
          properties: { chatId: { type: "string" } },
          required: ["chatId"],
        },
      },
      {
        name: "get_current_date",
        description: "Returns current date.",
      },
    ],
  },
];

// Basic guardrail check
function basicGuardrailCheck(text) {
  const bannedWords = ["hack", "password", "credit card", "address"];
  const isOffTopic = text.match(/weather|math|personal|politics/i);
  const hasBanned = bannedWords.some(word => text.toLowerCase().includes(word));
  if (isOffTopic) return { valid: false, message: "Sorry, I only handle IT support issues like laptop or Wi-Fi problems." };
  if (hasBanned) return { valid: false, message: "I can’t process requests involving sensitive or unsafe terms." };
  return { valid: true };
}

// Hono Route
app.post("/support", async (c) => {
  const { query, chatId, name, history = [] } = await c.req.json();
  if (!query || !chatId || !name) return c.json({ error: "Query, chatId, and name required" }, 400);

  const check = basicGuardrailCheck(query);
  if (!check.valid) return c.json({ response: check.message });

  try {
    const model = genAI.getGenerativeModel({
      model: "gemini-2.5-flash",
      tools,
      systemInstruction: `
        You are an IT support chatbot for ${name} (chatId: ${chatId}).
        - Only handle queries about IT issues (e.g., laptop, Wi-Fi, software, hardware). For off-topic queries, respond: "Sorry, I only handle IT support issues like laptop or Wi-Fi problems."
        - Never suggest harmful actions (e.g., hacking, unsafe hardware mods, data deletion without backups). If unsure, say: "Please consult a professional technician."
        - Avoid sensitive data (e.g., passwords, addresses, credit card info). If detected, respond: "I can’t process requests involving sensitive information."
        - Be polite, concise, professional, addressing user as ${name}.
        - For greetings like "Hi," call get_past_tickets to summarize past issues or ask: "Hi ${name}, how can I help with your IT issue today?"
        - For IT problems, provide a safe solution (e.g., "Try restarting your router for Wi-Fi issues"), then call create_ticket with chatId, problem summary, and solution.
        - If unclear, ask for clarification without creating a ticket.
        - Use history: ${JSON.stringify(history)} for context to personalize responses.
      `,
    });

    const chat = model.startChat({
      history: history.map(h => ({ role: h.isBot ? "assistant" : "user", parts: [{ text: h.message }] })),
    });
    let result = await chat.sendMessage(query);
    let response = result.response;
    let finalResponse = "";

    while (true) {
      const functionCalls = response.functionCalls();
      if (!functionCalls || functionCalls.length === 0) break;

      const functionResponses = [];
      for (const call of functionCalls) {
        const { name, args } = call;
        if (functions[name]) {
          const functionResult = await functions[name](args);
          if (functionResult.success) {
            finalResponse = functionResult.message || JSON.stringify(functionResult.tickets);
          } else {
            finalResponse = `Error: ${functionResult.error}`;
            break;
          }
          functionResponses.push({ functionResponse: { name, response: functionResult } });
        }
      }

      if (functionResponses.length > 0 && !finalResponse.startsWith("Error")) {
        result = await chat.sendMessage(functionResponses);
        response = result.response;
        if (!response.functionCalls()) {
          finalResponse = response.text() || finalResponse;
        }
      } else {
        break;
      }
    }

    if (!finalResponse && response.text()) {
      finalResponse = response.text();
    }

    const outputCheck = basicGuardrailCheck(finalResponse);
    if (!outputCheck.valid) return c.json({ response: outputCheck.message });

    return c.json({ response: finalResponse });
  } catch (error) {
    console.error("Error:", error);
    return c.json({ error: "Internal server error" }, 500);
  }
});

// Telegram Handler (Placeholder for WhatsApp)
bot.on("message", async (msg) => {
  if (msg.voice) return; // Voice not implemented yet
  const chatId = msg.chat.id.toString();
  const name = msg.from?.first_name || "User";
  const query = msg.text;
  if (!query) return;

  try {
    const history = await get_conversation_history(chatId);
    await save_message(chatId, query, false);
    const response = await fetch(`${process.env["PROD_URL"]}/support`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, chatId, name, history }),
    });
    const data = await response.json();
    const botResponse = data.response || "Sorry, I couldn’t process your request.";
    await bot.sendMessage(chatId, botResponse);
    await save_message(chatId, botResponse, true);
  } catch (error) {
    console.error("Error:", error);
    await bot.sendMessage(chatId, "An error occurred.");
  }
});

// WhatsApp Handler (Example with whatsapp-web.js, uncomment when ready)
// import WhatsApp from 'whatsapp-web.js';
// const { Client } = WhatsApp;
// const client = new Client({ authStrategy: new WhatsApp.LocalAuth() });
// client.on('ready', () => console.log('WhatsApp ready'));
// client.on('message', async (msg) => {
//   const chatId = msg.from; // Phone number as chatId
//   const name = msg._data.notifyName || "User";
//   const query = msg.body;
//   if (!query) return;
//   try {
//     const history = await get_conversation_history(chatId);
//     await save_message(chatId, query, false);
//     const response = await fetch(`${process.env["PROD_URL"]}/support`, {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({ query, chatId, name, history }),
//     });
//     const data = await response.json();
//     const botResponse = data.response || "Sorry, I couldn’t process your request.";
//     await msg.reply(botResponse);
//     await save_message(chatId, botResponse, true);
//   } catch (error) {
//     console.error("Error:", error);
//     await msg.reply("An error occurred.");
//   }
// });
// client.initialize();

console.log("Server running at http://localhost:3000");
export default app;