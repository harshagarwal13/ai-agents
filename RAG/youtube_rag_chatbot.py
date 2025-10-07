import re
import sys
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_nomic import NomicEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

# YouTube API imports
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
except ImportError:
    print("Error: youtube_transcript_api is not installed.")
    print("Please install it using: pip install youtube-transcript-api")
    sys.exit(1)


class YouTubeRAGChatbot:
    """A comprehensive YouTube RAG chatbot system."""

    def __init__(self):
        """Initialize the YouTube RAG chatbot."""
        self.embedding_function = NomicEmbeddings(model="nomic-embed-text-v1")
        self.llm = ChatGroq(model="llama-3.3-70b-versatile")
        self.vector_store = None
        self.retriever = None
        self.current_video_id = None
        self.current_video_title = "Unknown Video"
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        self.setup_workflow()

    def extract_video_id(self, url):
        """Extract video ID from YouTube URL or return the ID if already provided."""
        if url.startswith("http"):
            patterns = [
                r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)",
                r"youtube\.com\/watch\?.*v=([^&\n?#]+)",
            ]

            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    return match.group(1)

            print("Error: Could not extract video ID from URL")
            return None
        else:
            return url

    def get_youtube_transcript(self, video_input):
        """Get transcript for a YouTube video."""
        video_id = self.extract_video_id(video_input)
        if not video_id:
            return None, None

        try:
            print(f"Fetching transcript for video ID: {video_id}")

            # Create API instance and fetch transcript
            api = YouTubeTranscriptApi()
            fetched_transcript = api.fetch(video_id, languages=["en", "en-US", "en-GB"])

            # Combine all transcript text with timestamps
            full_transcript = ""
            transcript_with_timestamps = []

            for entry in fetched_transcript:
                text = entry.text.replace("\n", " ").strip()
                start_time = entry.start
                if text:
                    full_transcript += text + " "
                    transcript_with_timestamps.append(
                        {
                            "text": text,
                            "start": start_time,
                            "formatted_time": self.format_timestamp(start_time),
                        }
                    )

            return full_transcript.strip(), transcript_with_timestamps

        except TranscriptsDisabled:
            print(f"Error: Transcripts are disabled for video {video_id}")
            return None, None
        except NoTranscriptFound:
            print(f"Error: No transcript found for video {video_id}")
            return None, None
        except Exception as e:
            print(f"Error fetching transcript: {str(e)}")
            return None, None

    def format_timestamp(self, seconds):
        """Convert seconds to MM:SS format."""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def create_video_embeddings(self, video_input):
        """Create embeddings for a YouTube video transcript."""
        transcript, transcript_with_timestamps = self.get_youtube_transcript(
            video_input
        )

        if not transcript:
            return False

        video_id = self.extract_video_id(video_input)
        self.current_video_id = video_id

        # Split transcript into chunks
        chunks = self.text_splitter.split_text(transcript)

        # Create documents with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            # Find approximate timestamp for this chunk
            chunk_start = i * (len(transcript) // len(chunks)) if len(chunks) > 1 else 0
            approx_time = self.find_approximate_timestamp(
                chunk_start, transcript, transcript_with_timestamps
            )

            doc = Document(
                page_content=chunk,
                metadata={
                    "video_id": video_id,
                    "chunk_id": i,
                    "approximate_timestamp": approx_time,
                    "source": f"YouTube Video: {video_id}",
                },
            )
            documents.append(doc)

        # Create vector store
        self.vector_store = Chroma.from_documents(documents, self.embedding_function)
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 3}
        )

        print(
            f"✅ Created embeddings for {len(documents)} chunks from video {video_id}"
        )
        return True

    def find_approximate_timestamp(
        self, char_pos, full_transcript, transcript_with_timestamps
    ):
        """Find approximate timestamp for a character position in the transcript."""
        if not transcript_with_timestamps:
            return "00:00"

        # Simple approximation based on character position
        total_chars = len(full_transcript)
        if total_chars == 0:
            return "00:00"

        progress_ratio = char_pos / total_chars
        timestamp_index = int(progress_ratio * len(transcript_with_timestamps))
        timestamp_index = min(timestamp_index, len(transcript_with_timestamps) - 1)

        return transcript_with_timestamps[timestamp_index]["formatted_time"]

    def setup_workflow(self):
        """Setup the RAG workflow for YouTube videos."""

        # Define state
        class AgentState(TypedDict):
            messages: List[BaseMessage]
            documents: List[Document]
            on_topic: str
            rephrased_question: str
            proceed_to_generate: bool
            rephrase_count: int
            question: HumanMessage

        # Pydantic models for structured output
        class GradeQuestion(BaseModel):
            score: str = Field(
                description="Question is about the YouTube video content. If yes-> 'Yes' if no-> 'No'"
            )

        class GradeDocument(BaseModel):
            score: str = Field(
                description="Document is relevant to the question? If yes -> 'yes' if not-> 'No'"
            )

        # Node functions
        def question_rewriter(state: AgentState):
            print(f"🔄 Rewriting question...")

            # Reset state variables
            state["documents"] = []
            state["on_topic"] = ""
            state["rephrase_count"] = 0
            state["proceed_to_generate"] = False

            if "messages" not in state or state["messages"] is None:
                state["messages"] = []

            if state["question"] not in state["messages"]:
                state["messages"].append(state["question"])

            if len(state["messages"]) > 1:
                conversation = state["messages"][:-1]
                current_question = state["question"].content

                rewrite_prompt = ChatPromptTemplate.from_messages(
                    [
                        SystemMessage(
                            content="You are a helpful assistant that rephrases questions to be standalone and optimized for retrieving information from a YouTube video transcript."
                        ),
                        *conversation,
                        HumanMessage(
                            content=f"Rephrase this question to be standalone: {current_question}"
                        ),
                    ]
                )

                response = self.llm.invoke(rewrite_prompt.format_messages())
                better_question = response.content.strip()
                print(f"📝 Rephrased question: {better_question}")
                state["rephrased_question"] = better_question
            else:
                state["rephrased_question"] = state["question"].content

            return state

        def question_classifier(state: AgentState):
            print("🔍 Classifying question...")

            system_message = SystemMessage(
                content=f"""You are a classifier that determines whether a user's question is about the content of the YouTube video (ID: {self.current_video_id or "Unknown"}).

                If the question is asking about:
                - The video content, topics discussed, or information presented
                - What was said in the video
                - Summaries or explanations of the video
                - Any specific details from the video

                Respond with 'Yes'. Otherwise, respond with 'No'.
                """
            )

            human_message = HumanMessage(
                content=f"User Question: {state['rephrased_question']}"
            )
            grade_prompt = ChatPromptTemplate.from_messages(
                [system_message, human_message]
            )

            structured_llm = self.llm.with_structured_output(GradeQuestion)
            grader_llm = grade_prompt | structured_llm

            result = grader_llm.invoke({})
            state["on_topic"] = result.score.strip()
            print(f"📊 Question classification: {state['on_topic']}")
            return state

        def on_topic_router(state: AgentState):
            print("🧭 Routing based on topic...")
            on_topic = state.get("on_topic", "").strip().lower()
            if on_topic == "yes":
                return "retrieve"
            else:
                return "off_topic_response"

        def retrieve(state: AgentState):
            print("📚 Retrieving relevant video segments...")
            if not self.retriever:
                print("❌ No retriever available - video not processed")
                state["documents"] = []
                return state

            documents = self.retriever.invoke(state["rephrased_question"])
            print(f"📄 Retrieved {len(documents)} document chunks")
            state["documents"] = documents
            return state

        def retrieval_grader(state: AgentState):
            print("⚖️ Grading retrieved documents...")

            system_message = SystemMessage(
                content="""You are a grader assessing the relevance of retrieved video transcript segments to a user question.
                Only answer 'Yes' or 'No'.
                If the document contains information relevant to the user's question, respond with 'Yes'. Otherwise, respond with 'No'.
                """
            )

            structured_llm = self.llm.with_structured_output(GradeDocument)
            relevant_documents = []

            for doc in state["documents"]:
                human_message = HumanMessage(
                    content=f"User Question: {state['rephrased_question']}\n\nVideo transcript segment: {doc.page_content}"
                )
                grade_prompt = ChatPromptTemplate.from_messages(
                    [system_message, human_message]
                )
                grader_llm = grade_prompt | structured_llm

                result = grader_llm.invoke({})
                if result.score.strip().lower() == "yes":
                    relevant_documents.append(doc)

            state["documents"] = relevant_documents
            state["proceed_to_generate"] = len(relevant_documents) > 0
            print(f"✅ Found {len(relevant_documents)} relevant documents")
            return state

        def proceed_router(state: AgentState):
            rephrase_count = state.get("rephrase_count", 0)
            if state.get("proceed_to_generate", False):
                return "generate_answer"
            elif rephrase_count >= 2:
                return "cannot_answer"
            else:
                return "refine_question"

        def refine_question(state: AgentState):
            print("🔧 Refining question...")
            rephrase_count = state.get("rephrase_count", 0)
            if rephrase_count >= 2:
                return state

            question_to_refine = state["rephrased_question"]
            refine_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content="You are a helpful assistant that slightly refines questions to improve retrieval from YouTube video transcripts."
                    ),
                    HumanMessage(
                        content=f"Original question: {question_to_refine}\n\nProvide a slightly refined question that might work better for searching video content."
                    ),
                ]
            )

            response = self.llm.invoke(refine_prompt.format_messages())
            refined_question = response.content.strip()
            print(f"🔄 Refined question: {refined_question}")

            state["rephrased_question"] = refined_question
            state["rephrase_count"] = rephrase_count + 1
            return state

        def generate_answer(state: AgentState):
            print("💬 Generating answer...")

            if "messages" not in state or state["messages"] is None:
                state["messages"] = []

            history = state["messages"]
            documents = state["documents"]
            rephrased_question = state["rephrased_question"]

            # Format context with timestamps
            context_with_timestamps = []
            for doc in documents:
                timestamp = doc.metadata.get("approximate_timestamp", "00:00")
                context_with_timestamps.append(f"[{timestamp}] {doc.page_content}")

            context = "\n\n".join(context_with_timestamps)

            template = """
            Answer the question based on the following YouTube video transcript segments and chat history.

            Chat History: {history}

            Video Transcript Segments:
            {context}

            Question: {question}

            Please provide a helpful answer based on the video content. If you reference specific information,
            mention the approximate timestamp when relevant.
            """

            prompt = ChatPromptTemplate.from_template(template)
            response = self.llm.invoke(
                prompt.format_messages(
                    history=history, context=context, question=rephrased_question
                )
            )

            generation = response.content.strip()
            state["messages"].append(AIMessage(content=generation))
            print("✅ Answer generated successfully")
            return state

        def cannot_answer(state: AgentState):
            print("❌ Cannot answer question")
            if "messages" not in state:
                state["messages"] = []
            state["messages"].append(
                AIMessage(
                    content="I'm sorry, but I cannot find relevant information in the video transcript to answer your question."
                )
            )
            return state

        def off_topic_response(state: AgentState):
            print("🚫 Off-topic question")
            if "messages" not in state:
                state["messages"] = []
            state["messages"].append(
                AIMessage(
                    content=f"I can only answer questions about the content of the YouTube video (ID: {self.current_video_id or 'Unknown'}). Please ask something related to what was discussed in the video."
                )
            )
            return state

        # Build workflow
        checkpointer = MemorySaver()

        workflow = StateGraph(AgentState)
        workflow.add_node("question_rewriter", question_rewriter)
        workflow.add_node("question_classifier", question_classifier)
        workflow.add_node("off_topic_response", off_topic_response)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("retrieval_grader", retrieval_grader)
        workflow.add_node("generate_answer", generate_answer)
        workflow.add_node("refine_question", refine_question)
        workflow.add_node("cannot_answer", cannot_answer)

        workflow.add_edge("question_rewriter", "question_classifier")
        workflow.add_conditional_edges(
            "question_classifier",
            on_topic_router,
            {
                "retrieve": "retrieve",
                "off_topic_response": "off_topic_response",
            },
        )
        workflow.add_edge("retrieve", "retrieval_grader")
        workflow.add_conditional_edges(
            "retrieval_grader",
            proceed_router,
            {
                "generate_answer": "generate_answer",
                "refine_question": "refine_question",
                "cannot_answer": "cannot_answer",
            },
        )
        workflow.add_edge("refine_question", "retrieve")
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("cannot_answer", END)
        workflow.add_edge("off_topic_response", END)
        workflow.set_entry_point("question_rewriter")

        self.graph = workflow.compile(checkpointer=checkpointer)
        self.AgentState = AgentState

    def ask_question(self, question, thread_id="default"):
        """Ask a question about the loaded YouTube video."""
        if not self.vector_store:
            return (
                "❌ No YouTube video has been processed yet. Please load a video first."
            )

        input_data = {"question": HumanMessage(content=question)}
        config = {"configurable": {"thread_id": thread_id}}

        try:
            result = self.graph.invoke(input=input_data, config=config)
            return result["messages"][-1].content
        except Exception as e:
            return f"Error processing question: {str(e)}"

    def run_interactive_chat(self):
        """Run an interactive chat session."""
        print("🎥 YouTube RAG Chatbot")
        print("=" * 50)

        # Get YouTube video
        print("\nFirst, let's load a YouTube video.")
        video_input = input("Enter YouTube URL or video ID: ").strip()

        if not video_input:
            print("❌ No video provided. Exiting.")
            return

        print(f"\n🔄 Processing video: {video_input}")
        if not self.create_video_embeddings(video_input):
            print("❌ Failed to process video. Exiting.")
            return

        print(f"\n✅ Video processed successfully!")
        print(f"📹 Video ID: {self.current_video_id}")
        print("\n💬 You can now ask questions about the video content.")
        print("Type 'quit', 'exit', or 'new video' to change options.\n")

        thread_id = "interactive_session"

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit"]:
                    print("👋 Goodbye!")
                    break

                if user_input.lower() == "new video":
                    print("\n🎥 Loading new video...")
                    video_input = input("Enter YouTube URL or video ID: ").strip()
                    if video_input:
                        if self.create_video_embeddings(video_input):
                            print(f"✅ New video loaded: {self.current_video_id}")
                            thread_id = f"session_{self.current_video_id}"  # New thread for new video
                        else:
                            print("❌ Failed to load new video.")
                    continue

                print("🤖 Assistant: ", end="", flush=True)
                response = self.ask_question(user_input, thread_id)
                print(response)
                print()

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function to run the YouTube RAG chatbot."""
    try:
        chatbot = YouTubeRAGChatbot()
        chatbot.run_interactive_chat()
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        print("Make sure you have set up your environment variables (.env file) with:")
        print("- GROQ_API_KEY")
        print("- NOMIC_API_KEY")


if __name__ == "__main__":
    main()
