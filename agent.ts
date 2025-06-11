import { MemorySaver } from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { ChatOpenAI } from "@langchain/openai";
import { TavilySearch } from "@langchain/tavily";
import "dotenv/config";

// Initialize the TavilySearch tool with a maximum of 3 results
const agentTools = [new TavilySearch({ maxResults: 3 })];

// Initialize the OpenAI model with the specified parameters
const agentModel = new ChatOpenAI({
  model: "gpt-4.1-nano",
  temperature: 0,
  openAIApiKey: process.env.OPENAI_API_KEY,
});

// Create a memory saver to store the agent's state
const memory = new MemorySaver();

// Create the agent with the model, tools, and memory saver
const agent = await createReactAgent({
  llm: agentModel,
  tools: agentTools,
  checkpointSaver: memory,
});

// Create a unique thread ID for identifying that conversation
const config = { configurable: { thread_id: "42" } };

// First interaction with the agent
const input = {
  role: "user",
  content: "What is the current weather in kathmandu?",
};
const output = await agent.invoke({ messages: input }, config);

console.log(output?.messages?.[output.messages.length - 1]?.content);

// Second interaction with the agent, using the same thread ID
const input2 = { role: "user", content: "What is about biratnagar?" };
const output2 = await agent.invoke({ messages: input2 }, config);

console.log(output2?.messages?.[output2.messages.length - 1]?.content);
