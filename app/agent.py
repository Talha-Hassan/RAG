from typing import Optional
from langchain_protocol import TypedDict
from typing_extensions import Annotated
from langgraph.graph import StateGraph , START , END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage , AIMessage , BaseMessage
from langsmith import traceable


from app.config import get_settings


class AgentState(TypedDict):
    """State of the agent, including the graph and any other relevant information."""
    messages : Annotated[list[BaseMessage], add_messages]
    err: Optional[str]
    retry_count: int
    model_used: str


class ProductionAgent:
    settings = get_settings()

    def __init__(self):
        self.primary_model =  ChatOpenAI(model_name=self.settings.primary_model, temperature=0 , max_retries=0 , timeout=30, api_key=self.settings.openai_api_key)
        self.max_retries = self.settings.max_retries
        self.graph = self._build_graph()
    
    def _build_graph(self):

        def process_input(state: AgentState) -> dict:
            """Process the input messages and generate a response."""
            try:
                response = self.primary_model.invoke(state['messages'])
                return {
                    "messages": [response],
                    "err": None,
                    "model_used": self.settings.primary_model,
                }
            except Exception as e:
                return {
                    "err" : str(e),
                    "model_used": self.settings.primary_model,
                    "retry_count": state['retry_count'] + 1
                }
        
        def handle_error(state: AgentState) -> dict:
            return {
                "messages": [AIMessage(content=f"I am Sorry!")],
                "model_used": "error_handler"
            }
    
        def route_after_process(state: AgentState) -> str:
            if state['err'] is None:
                return "done"
            elif state['retry_count'] < self.max_retries:
                return "start"
            else:
                return "error"
        
        graph = StateGraph(AgentState)

        graph.add_node("process_input", process_input)
        graph.add_conditional_edges("process_input", route_after_process, {
            "done": END,
            "start": "process_input",
            "error": "error_handler"
        })
        graph.add_node("error_handler", handle_error)
        graph.add_edge(START, "process_input")
        graph.add_edge("error_handler", END)

        return graph.compile()
    

    async def invoke(self, message: str) -> str:
        """Invoke the agent with a user message and return the response."""
        initial_state: AgentState = {
            "messages": [HumanMessage(content=message)],
            "err": None,
            "retry_count": 0,
            "model_used": self.settings.primary_model
        }
        # print(f"Invoking agent with initial state: {initial_state}")
        final_state = self.graph.invoke(initial_state)

        # print(f"Messages in final state: {final_state['err']}")
        
        return {
            "response": final_state['messages'][1].content if final_state['messages'] else "No response generated.",
            "model_used": final_state['model_used'],
            "error": final_state['err']
        }