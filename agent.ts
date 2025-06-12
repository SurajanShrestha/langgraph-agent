import { MessagesAnnotation, START, StateGraph } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { ChatOpenAI } from "@langchain/openai";
import { TavilySearch } from "@langchain/tavily";
import "dotenv/config";
import { AIMessage } from "@langchain/core/messages";

// NOTE 💡: INDUSTRY STANDARD WAY OF CREATING AGENT WITH FINE GRAINED CONTROL USING STATE GRAPH

// Initialize the TavilySearch tool with a maximum of 3 results
const agentTools = [new TavilySearch({ maxResults: 3 })];
const toolNode = new ToolNode(agentTools);

// Initialize the OpenAI model with the specified parameters
const agentModel = new ChatOpenAI({
  model: "gpt-4.1-nano",
  temperature: 0,
  openAIApiKey: process.env.OPENAI_API_KEY,
}).bindTools(agentTools);

// Define the function that determines the whether to continue or not
const shouldContinue = ({ messages }: typeof MessagesAnnotation.State) => {
  const lastMessage = messages[messages.length - 1] as AIMessage;
  // If the LLM makes a tool call, then we route to the "tools" node
  if (lastMessage.tool_calls?.length) {
    return "tools";
  }
  // Otherwise, we stop (reply to the user) using the special "__end__" node
  return "__end__";
};

// Define the function that calls the model
const callModel = async (state: typeof MessagesAnnotation.State) => {
  const response = await agentModel.invoke(state.messages);
  // We return a list, because this will get added to the existing list
  return { messages: [response] };
};

// Define a new graph
// START --> agent  <-- tools
// agent --> shouldContinue (YES) --> tools
// agent --> shouldContinue (NO) --> END
const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addEdge(START, "agent")
  .addNode("tools", toolNode)
  .addEdge("tools", "agent")
  .addConditionalEdges("agent", shouldContinue);

// Compile it to a a Langchain Runnable
const app = workflow.compile();

// Use the agent
const finalState = await app.invoke({
  messages: { role: "user", content: "What is the weather in Kathmandu?" },
});
console.log(finalState?.messages?.[finalState.messages.length - 1]?.content);

// IMP 💡: WE SHOULD'VE USED "MemorySaver()" INSTEAD OF APPENDING PREVIOUS MESSAGES.
// HERE, WE APPEND BECAUSE THIS LESSON TEACHES THE RAW MECHANICS OF LANGGRAPH
// BUT YES, IT SACRIFICES DEVELOPER ERGONOMICS FOR CLARITY
// IF BUILDING A REAL AGENT, USE "MemorySaver()" TO STORE THE STATE (DONE IN chatbot-llm project)
const nextState = await app.invoke({
  messages: [
    ...finalState?.messages,
    { role: "user", content: "What about Biratnagar?" },
  ],
});
console.log(nextState?.messages?.[nextState.messages.length - 1]?.content);
