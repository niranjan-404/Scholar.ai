from docling.document_converter import DocumentConverter
from langchain.text_splitter import MarkdownHeaderTextSplitter 
import fitz 
import docx
from io import BytesIO
from PIL import Image
import os
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader
from support_functions.get_token_count import num_tokens_from_string
from load_models.model_configurations import *
from langchain_core.output_parsers import BaseOutputParser
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain.tools.base import StructuredTool
from langchain_core.output_parsers import BaseOutputParser
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.messages import AIMessage, HumanMessage
import base64
from support_functions.mongodb_hybrid_search import mongodb_hybrid_search
from support_functions.load_db_client import mongo_client
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langchain.agents import create_openai_functions_agent
from mimetypes import guess_type
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,)
from langchain.schema.document import Document
from datetime import timezone,date
from datetime import datetime
from pathlib import Path
from typing import List
from langchain.schema import Document
import fitz  
from docling.document_converter import DocumentConverter
from langchain.text_splitter import MarkdownHeaderTextSplitter  
import tempfile
import os
from pathlib import Path
import operator
from support_functions.create_mongo_index import *

context_limit = 70000

class DocumentProcessing:

        def __init__(self,file_path,st) -> None:
            self.azureopenai_client = load_azureopenai_client()
            self.primary_llm =  load_primary_llm()
            self.llm = load_secondary_llm()
            self.embedding_model = load_embedding_model()
            self.st = st
            self.file_path = file_path
            self.collection = mongo_client["TEXT_BOOKS"]["DOCUMENT_EMBEDDINGS"]
            self.recommended_questions = []

        def create_vectorStore(self,docs):
                
                self.collection.delete_many({})

                MongoDBAtlasVectorSearch.from_documents(docs, self.embedding_model, collection=self.collection)

                indexes = self.collection.list_search_indexes()
                vector_search_index_flag = False
                keyword_search_index_flag = False
                
                for i in indexes:
                        if i["name"] == "vector_search_index":
                                vector_search_index_flag = True
                        if i["name"] == "keyword_search_index":
                                keyword_search_index_flag = True     

                if vector_search_index_flag == False:
                        create_vector_index(self.collection)
                
                if keyword_search_index_flag == False:
                        create_keyword_search_index(self.collection)

        def get_pdf_page_count(self):
                doc = fitz.open(self.file_path)
                return len(doc) 

        def is_within_context_length(self,docs):
            if type(docs) != str: 
                    text = ""
                    for page in docs:
                            text = text +"\n"+str(page.page_content).replace("\n"," ")
                    
                    if num_tokens_from_string(text)<context_limit:
                            return True
                    else:
                            return False
            else:
                    if num_tokens_from_string(docs)<context_limit:
                            return True
                    else:
                            return False

        def extract_table_of_contents(self,text_input):
            
            writer_prompt = PromptTemplate(input_variables=["text_input"],
                    template= '''You are a specialized document structure analyzer focused on extracting and formatting tables of contents into clean markdown. Your task is to:

                        1. Analyze the provided document text and identify the hierarchical structure:
                        - Main chapter titles and page numbers
                        - Nested subheadings (e.g., 1.1, 1.1.2) with page numbers
                        - Front matter (preface, acknowledgments)
                        - Back matter (appendices, glossary, index)
                        - Special sections or supplementary content

                        2. Format the content hierarchy in markdown using:
                        - # for main chapters
                        - ## for first-level subheadings 
                        - ### for second-level subheadings
                        - #### for third-level subheadings
                        - Preserve original numbering schemes within the headers
                        - Right-align page numbers using markdown table syntax

                        3. Structure the output as:
                        ```markdown
                        # Chapter 1: [Title] ........................... [page]
                        ## 1.1 [Subtitle] ............................. [page]
                        ### 1.1.1 [Sub-subtitle] ...................... [page]

                        Portion of text extracted from test book:
                        {text_input}
                    ''')
            
            if not isinstance(text_input, str):
                text_input = str(text_input)

            chain = writer_prompt | self.llm  | StrOutputParser()
            output_text = chain.invoke({"text_input":text_input})
            
            self.st.subheader("📌 Extracted Table of content:")
            
            self.st.success(output_text)

            return output_text

        def load_markdown_document(self):
                converter = DocumentConverter()
                doc = fitz.open(self.file_path)
                headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
                markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
                file_name = Path(self.file_path).name

                parsed_docs = []

                for page_number, page in enumerate(doc, start=1):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                temp_page_path = tmp_file.name

                        page_document = fitz.open()  
                        page_document.insert_pdf(doc, from_page=page_number - 1, to_page=page_number - 1)
                        page_document.save(temp_page_path)
                        page_document.close()

                        conversion_result = converter.convert(temp_page_path)
                        page_markdown = conversion_result.document.export_to_markdown()

                        os.remove(temp_page_path)

                        markdown_sections = markdown_splitter.split_text(page_markdown)

                        for section in markdown_sections:
                                for key_ in section.metadata.keys():
                                        if key_ in ["Header 2","Header 1"]: 
                                                section.page_content = f"{section.metadata[key_]} \n {section.page_content}"

                        section.metadata["page_number"] = page_number
                        section.metadata["source"] = file_name
                        
                        parsed_docs.append(section)

                return parsed_docs

        def load_langchain_doc(self):
            if self.file_path.endswith('.pdf'):
                loader = PyPDFLoader(self.file_path)
                docs = loader.load()
                        
            elif self.file_path.endswith('.docx') or self.file_path.endswith('.doc'):
                loader = Docx2txtLoader(self.file_path)
                docs = loader.load()
                    
            elif self.file_path.endswith('.txt') or self.file_path.endswith('.md'):
                loader = TextLoader(self.file_path,encoding = 'UTF-8')
                docs = loader.load()

            return docs

        def extract_images_from_file(self, page_number, output_dir="extracted_images"):
            """
            Extracts all images from a specified page of a PDF or DOCX file.
            
            Parameters:
            - file_path: str, path to the input file (PDF or DOCX)
            - page_number: int, the page number to extract images from (1-based index)
            - output_dir: str, directory where extracted images will be saved (default: 'extracted_images')

            Returns:
            - List of extracted image file paths.
            """
            os.makedirs(output_dir, exist_ok=True)
            extracted_images = []

            if self.file_path.lower().endswith(".pdf"):
                doc = fitz.open(self.file_path)
                if page_number < 1 or page_number > len(doc):
                    raise ValueError("Invalid page number for PDF.")
                
                page = doc[page_number - 1] 
                images = page.get_images(full=True)

                for i, img in enumerate(images):
                    xref = img[0]  
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    image_path = os.path.join(output_dir, f"pdf_page{page_number}_img{i}.{image_ext}")
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    extracted_images.append(image_path)

            elif self.file_path.lower().endswith(".docx"):
                doc = docx.Document(self.file_path)
                images = doc.inline_shapes

                if page_number != 1:
                    raise ValueError("DOCX files do not have pages like PDFs. Extracting from the whole document.")

                for i, shape in enumerate(images):
                    if shape._inline.graphic.graphicData.pic.blipFill:
                        blip = shape._inline.graphic.graphicData.pic.blipFill.blip
                        rID = blip.embed
                        image_part = doc.part.related_parts[rID]
                        image_bytes = image_part.blob

                        image = Image.open(BytesIO(image_bytes))
                        image_ext = image.format.lower()  

                        image_path = os.path.join(output_dir, f"docx_img{i}.{image_ext}")
                        image.save(image_path)
                        extracted_images.append(image_path)

            else:
                raise ValueError("Unsupported file format. Please provide a PDF or DOCX file.")

            return extracted_images

        def generate_notes(self,starting_page=None,ending_page=None,topic=None):
                
                def get_notes(docs,user_query):
                        extract_notes_prompt = PromptTemplate(input_variables=["docs","user_query"],
                        template= '''As an expert academic note-taker, create detailed, structured, and easy-to-understand notes from the provided document for students. The notes should focus on the specific topic mentioned in the user query, while minimizing unrelated content. Ensure the notes are concise, student-friendly, and highlight the most important points as they would appear in a textbook.

                        Guidelines for Note-Taking:

                        1. Document Overview  
                        - Identify the document type (e.g., research paper, textbook chapter, lecture notes, etc.).  
                        - Summarize the main topic and purpose of the document.  
                        - Address the user's specific requirements (if provided) to tailor the notes accordingly.  

                        2. Focus on Specific Topic (if mentioned in user query):  
                        - Concentrate exclusively on the topic specified in the user query.  
                        - Extract and explain all relevant content related to the topic, including:  
                        - Core concepts, theories, and definitions.  
                        - Key arguments, findings, or conclusions.  
                        - Examples, analogies, or diagrams (if applicable).  
                        - Ignore or minimize any content in the document that is not directly related to the specified topic, unless it provides essential foundational context.  

                        3. Structure & Organization  
                        - Organize notes in a logical, point-wise format with clear headings and subheadings.  
                        - Use bullet points, numbered lists, or tables for better readability.  
                        - Maintain a hierarchical structure (e.g., main ideas → supporting details).  

                        4. Key Content Extraction 
                        - Extract and explain core concepts, theories, and definitions in simple terms.  
                        - Highlight key arguments, findings, or conclusions from the document.  
                        - Include examples, analogies, or diagrams (if applicable) to clarify complex ideas.  

                        5. Academic Standards 
                        - Preserve citations, references, and technical terms accurately.  
                        - Include formulas, equations, or methodologies with clear explanations.  
                        - Ensure the notes adhere to academic precision and are suitable for students.  

                        ---

                        Response Format:

                        - If a specific topic is mentioned in the user query:  
                        - Focus exclusively on the requested topic.  
                        - Provide detailed explanations of all relevant content.  
                        - Ignore or briefly summarize unrelated content unless it provides essential context.  
                        - Add cross-references to related concepts or sections for better understanding.  

                        - If no specific topic is mentioned:  
                        - Provide comprehensive coverage of the document.  
                        - Balance depth vs. breadth to include all significant points.  
                        - Ensure the notes are self-contained and easy to follow.  

                        ---

                        Input Details:

                        - Document: 
                        ```  
                        {docs}  
                        ```  

                        - User Requirements:
                        ```  
                        {user_query}  
                        ```  

                        ---

                        Final Output Requirements:  
                        - Ensure the notes are accurate, complete, and academically precise.  
                        - Use bold or *italics* to emphasize key terms or concepts.  
                        - Include headings, subheadings, and bullet points for better organization.  
                        - Make the notes student-friendly, focusing on the most important points as they would appear in a textbook.  
                        - Output should be clean notes since it is directly presented to students
                        ''')

                        parser = StrOutputParser()
                        extract_notes_chain = extract_notes_prompt | self.primary_llm | parser
                        notes = extract_notes_chain.invoke({"docs":docs,"user_query":user_query})
                        try:
                               questions = self.generate_questions(docs)
                               self.st.session_state.recommended_questions = questions
                               self.recommended_questions = questions
                        except:
                               pass

                        return notes
            
                if not topic:

                        if starting_page or ending_page:

                                if not starting_page:
                                        starting_page = 1
                                elif not ending_page:
                                        ending_page = self.st.session_state.document_details["total_pages"]

                                if starting_page!= ending_page:
                                        query = {"page_number": {"$gte": starting_page, "$lte": ending_page}}
                                        documents = list(self.collection.find(query))
                                else:
                                        query = {"page_number": ending_page}
                                        documents = list(self.collection.find(query))

                        docs ="\n ".join(doc["text"] for doc in documents if "text" in doc)
                        notes = get_notes(user_query="Provide a overall notes",docs=docs)

                elif topic:
                        docs = mongodb_hybrid_search(query=topic,top_k=6,collection=self.collection)
                        notes = get_notes(user_query=topic,docs=docs)
                

                return notes

        def generate_questions(self,docs):
               
                generate_questions_prompt = PromptTemplate(input_variables=["docs","user_query"],
                template= '''You are a question recommendation assistant in an academic setting, working with provided textbook documents. Your task is to recommend relevant questions for users to ask.

                To generate valid questions, follow these steps:
                - Identify the main topic of the document.
                - Extract key information, focusing on essential details while omitting extraneous content.
                - Use this information to generate 1-3 relevant and meaningful questions that user cans ask about this document
                
                Output Format:
                Return a valid dictionary with the key "questions", where the value is a list of the generated questions. 

               
               Document
                {docs}  

                
                Your task is to create questions based on the above document.  


                Return a valid dictionary with the key "questions"
                ''')


                parser = JsonOutputParser()
                generate_questions_chain = generate_questions_prompt | self.llm | parser
                questions = generate_questions_chain.invoke({"docs":docs})
                
                if type(questions) == list:
                        return questions[0].get("questions")
                
                return questions.get("questions")
               
        def chat_with_document(self,chat_history, user_question,table_of_contents):  

                def image_to_base64(image_path):
                        mime_type, _ = guess_type(image_path)
                        if mime_type is None:
                                mime_type = 'application/octet-stream'  
                        with open(image_path, "rb") as image_file:
                                base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
                        return f"data:{mime_type};base64,{base64_encoded_data}"

                def qa_with_image(page_number:int,user_query:str):
                        """Use this tool if the user provides a specific page number and asks for an explanation or has a question about the images on that page. The tool's input should include `page_number`, indicating the page whose images need to be analyzed, and `user_query`, representing the user's question."""

                        content= [ { "type": "text", "text": user_query }]

                        extracted_images = self.extract_images_from_file(page_number=page_number)

                        if extracted_images:
                                for image in extracted_images:
                                        image64 = image_to_base64(image)

                                        content.append({"type": "image_url","image_url": {"url": image64}})

                                        try:
                                                os.remove(image)
                                        except:
                                               pass

                                deployment_name = "gpt-4o"
                                response = self.azureopenai_client.chat.completions.create(
                                model=deployment_name,
                                messages=[{ "role": "system", "content": "You are a helpful assistant for image-based question answering in an academic setting"},
                                        {"role": "user", "content":content}]).choices[0].message.content

                                return response
                        else:
                               return f"No image was found on page {page_number}. Please provide the exact page number where the image is located."
                                               
                def answer_question_from_document(user_query):
                        """This tool is designed to assist users by providing answers to their queries when no specific page number or reference is provided. If a user asks a question without specifying a page number or source, this tool should be the first resource utilized to generate a response"""        
                        
                        class GraphState(TypedDict):
                                """
                                Represents the state of our graph.
                                Attributes:
                                        question: question
                                        generation: llm response 
                                        documents: list of context documents 
                                        rephrase_question: wether the question need to be rephrased or not for better retrieval
                                        chat_history : Previous user interactions
                                        is_question: wether provided question is question or not 
                                """
                                question: str
                                generation: str
                                documents: List[str]
                                rephrase_question: str
                                chat_history : str
                                is_question : bool
                                retrieval_count : int
                                is_known_lang : str

                        class LineListOutputParser(BaseOutputParser[List[str]]):

                                def parse(self, text: str) -> List[str]:
                                        lines = list(text.strip().split("\n"))
                                        return lines
                        
                        def retrieve(state):
                                """
                                Retrieve documents
                                """
                                question = state["question"]
                                docs = mongodb_hybrid_search(question,top_k=4,collection=self.collection)
                                if type(state.get("retrieval_count")) != int:
                                        state["retrieval_count"] = 0

                                retrieval_count = state["retrieval_count"] + 1
                                state["retrieval_count"] = retrieval_count

                                return {"documents": docs, "question": question,"retrieval_count":retrieval_count}

                        def grade_documents(state):
                                """
                                Determines whether the retrieved documents are relevant to the question
                                by using an llm Grader.
                                If any document are not relevant to question or documents are empty 
                                Helps filtering out irrelevant documents
                                """
                                question = state["question"]
                                documents = state["documents"]

                                SYS_PROMPT = """You are an expert grader assessing relevance of a retrieved document to a user question.
                                                Follow these instructions for grading:
                                                - If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
                                                - If the document contains indirect information that can help answer the question, grade it as relevant.
                                                - Your grade should be either 'Yes' or 'No' to indicate whether the document is relevant to the question or not.
                                                
                                                The output must be a json dict object with key `grade` and value is the grade
                                                ["grade":"Yes"] or ["grade":"No"]

                                                Output should be a valid json dict
                                                """

                                grade_prompt = ChatPromptTemplate.from_messages(
                                        [("system", SYS_PROMPT),
                                        ("human", """Retrieved document:
                                                        {document}
                                                User question:
                                                        {question}"""),])
                                
                                doc_grader = (grade_prompt|self.llm|JsonOutputParser())

                                doc_grader_using_flexi_primary_llm = (grade_prompt|self.primary_llm|JsonOutputParser())

                                filtered_docs = []
                                rephrase_question = "Yes"
                                if documents:
                                        for d in documents:
                                                try:
                                                        score = doc_grader.invoke({"question": question, "document": d.page_content})
                                                        grade = score["grade"]
                                                        if grade == "Yes":
                                                                filtered_docs.append(d)
                                                                rephrase_question = "No"
                                                except:
                                                        if self.compliance_preference == 1:
                                                                try:
                                                                        score = doc_grader_using_flexi_primary_llm.invoke({"question": question, "document": d.page_content})
                                                                        grade = score["grade"]
                                                                        if grade == "Yes":
                                                                                filtered_docs.append(d)
                                                                                rephrase_question = "No"
                                                                except:
                                                                        pass

                                                        else:
                                                                pass
                                                
                                return {"documents": filtered_docs, "question": question , "rephrase_question":rephrase_question}

                        def rewrite_query(state):
                                """
                                Rewrite the query to produce a better question.
                                """
                                def unique_documents(documents) -> List[Document]:
                                        return [doc for i, doc in enumerate(documents) if doc not in documents[:i]]

                                output_parser = LineListOutputParser()


                                QUERY_PROMPT = PromptTemplate(
                                input_variables=["question"],

                                template="""You are an AI assistant tasked with enhancing document retrieval from a vector database. Generate 5 alternative versions of the given user question, each on a new line. These variations should:

                                1. Explore different angles and perspectives on the original query
                                2. Use synonyms and related concepts to broaden the search
                                3. Break down complex questions into simpler components
                                4. Generalize specific queries to capture relevant broader topics
                                5. Specify or narrow down broad queries for more focused results

                                --> Additionally Create two out of five queries that indirectly target core information, which can help answer the user's question. This should be designed to retrieve key data that, when combined or inferred, indirectly provides an answer.
                                        - Example 1:
                                                User's Question: 'Is a particular event over?'
                                                output:
                                                'What is the event date?'
                                                'When will the event ends?'
                                        - Example 2:
                                                User's Question: 'Has JOHN turned 18?'
                                                output:
                                                'What is JOHN's date of birth?'
                                                'When was JOHN born?'

                                - Avoid repetition and ensure each version is distinct
                                - Maintain the original intent and key elements of the user's question
                                - Adapt the language complexity to match the original query's style

                                
                                Provide these alternative questions separated by newlines.

                                Original question: '{question}'""",
                                )

                                llm_chain = QUERY_PROMPT | self.primary_llm | output_parser
                                
                                vectordb = MongoDBAtlasVectorSearch(collection = self.collection, embedding = self.embedding_model ,index_name="vector_search_index" , text_key = "text", embedding_key = "embedding")

                                retriever=vectordb.as_retriever()

                                queries = llm_chain.invoke(state["question"])

                                document_lists = []

                                for query in queries:
                                        if query != "" or query != " ":
                                                try:
                                                        docs = retriever.invoke(query)
                                                        document_lists.extend(docs)
                                                except:
                                                        pass

                                docs = unique_documents(document_lists)      

                                return {"documents": docs}

                        def generate_answer(state):
                                """
                                Generate answer from context document using llm
                                """
                                self.st.session_state["recommended_questions"] = []
                                question = state["question"]
                                documents = state["documents"]

                                today = date.today()
                                day_num = datetime.today().weekday()
                                days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday",]
                                weekday = days[day_num]

                                template = """System: You are a precise academic assistant specializing in answering student questions based on authoritative source materials. Current date: {weekday}, {today}.

                                Query: {question}

                                Response Guidelines:
                                1. Answer solely from the provided reference material
                                2. Maximum length: 3-4 concise sentences
                                3. Use academic tone and precise language
                                4. For better understanding, if examples are required and available in the provided document, use them to help the student better grasp the topic.
                                4. For numerical data, include specific figures
                                5. If information is absent, state: "The provided reference material does not contain information about [specific aspect asked]"

                                Citation Format:
                                - End each answer with: Reference: [filename], p.[page_number]
                                - For multiple sources, separate with semicolons

                                Constraints:
                                - No hypotheticals or generalizations
                                - Maintain verbatim accuracy to sources
                                - If asked about trends/patterns, only describe what's explicitly stated
                                - For ambiguous queries, request clarification before answering
                                - If information is not available in the Reference Material to answer user query just state you don't have information to answer that query
                                
                                Reference Material:
                                ```
                                {context}
                                ```
                                
                                """
                                
                                prompt = PromptTemplate(input_variables=["question","context"],template=template)

                                extracted_docs = [Document(page_content=doc.page_content,
                                        metadata={
                                        "file": (doc.metadata["source"].get("file", "") if isinstance(doc.metadata["source"], dict) else ""),
                                        "page_number": doc.metadata["source"].get("page_number", "") if isinstance(doc.metadata["source"], dict) else ""
                                        }
                                ) for doc in state["documents"]]

                                rag_chain = prompt | self.primary_llm | StrOutputParser()

                                try:
                                        questions = self.generate_questions(extracted_docs)
                                        self.st.session_state.recommended_questions = questions
                                        self.recommended_questions = questions
                                except:
                                        pass

                                result = rag_chain.invoke({"question":user_query,"context":extracted_docs,"today":today,"weekday":weekday})

                                return {"documents": documents, "question": question, "generation": result}

                        def generate_or_rewrite_query(state):
                                if state["rephrase_question"] == "Yes" and state["retrieval_count"] == 1:
                                        return "rewrite_query"
                                else:
                                        return "generate_answer"
                        
                        agentic_rag = StateGraph(GraphState)

                        agentic_rag.add_node("retrieve", retrieve)                
                        agentic_rag.add_node("grade_documents", grade_documents)  
                        agentic_rag.add_node("rewrite_query", rewrite_query)     
                        agentic_rag.add_node("generate_answer", generate_answer)  

                        agentic_rag.set_entry_point("retrieve")
                        agentic_rag.add_edge("retrieve", "grade_documents")
                        agentic_rag.add_conditional_edges("grade_documents",generate_or_rewrite_query,
                                                        {"rewrite_query": "rewrite_query", "generate_answer": "generate_answer"},)
                        agentic_rag.add_edge("rewrite_query", "generate_answer")
                        agentic_rag.add_edge("generate_answer", END)

                        agentic_rag = agentic_rag.compile()

                        response = agentic_rag.invoke({"question":user_query})
                        
                        return response["generation"]
                        
                def user_interaction(message):
                    """Use this tool to request clarification from the user when their query is ambiguous, incomplete, or requires additional details. The input must be the question you want to ask the user, and the output will be their response"""

                    with self.st.form("my_form"):

                        user_text = self.st.text_input(message)

                        submitted = self.st.form_submit_button("Send")

                    if submitted:

                        return response

                def summarize_document(user_query,starting_page,ending_page):
                        """Use this tool to summarize the document based on the user's query and preferences. The input includes:user_query: The specific aspect the user wants summarized ,user's preference: The preferred style or format of the summary. starting_page and ending_page: Define the page range to be summarized. If the user wants a summary of a single page, set both starting_page and ending_page to the same value"""

                        def get_summary(docs,user_query):
                                summary_prompt = PromptTemplate(input_variables=["docs","user_query"],
                                template= ''' You have to create a detailed summary from the given document

                                To generate a detailed summary of the document, consider the following steps:

                                * Begin by identifying the document title to establish context.
                                * Extract key information focusing on essential details while omitting extraneous information.
                                * Ensure that the summary accurately represents the provided document's main topic, objectives, and key points to provide readers with a comprehensive understanding.
                                
                                document = {docs}

                                You need to summarize the above document in accordance with the requirements outlined in the following user query.

                                user query = {user_query}

                                If no specific requirements are mentioned in the provided user query, then you should give a detailed summary that includes each and every important point mentioned in the provided document        
                                ''')


                                parser = StrOutputParser()
                                summary_chain = summary_prompt | self.primary_llm | parser
                                summary = summary_chain.invoke({"docs":docs,"user_query":user_query})
                                return summary
                        
                        def get_dynamic_summary(docs, user_query=None):
                                
                                if self.is_within_context_length(docs):
                                        return get_summary(docs, user_query)

                                mid = len(docs) // 2
                                left_part = docs[:mid]
                                right_part = docs[mid:]

                                left_summary = get_dynamic_summary(left_part, user_query)
                                right_summary = get_dynamic_summary(right_part, user_query)

                                return get_summary(left_summary + " " + right_summary, user_query)

                        if starting_page or ending_page:

                                if not starting_page:
                                        starting_page = 1
                                elif not ending_page:
                                        ending_page = self.st.session_state.document_details["total_pages"]

                                if starting_page != ending_page:
                                        query = {"page_number": {"$gte": starting_page, "$lte": ending_page}}
                                        documents = self.collection.find(query)
                                else:
                                        query = {"page_number": starting_page}
                                        documents = list(self.collection.find(query))
                        
                        else:
                                documents = list(self.collection.find({}))
                        
                        docs = "\n ".join(doc["text"] for doc in documents if "text" in doc)
                        
                        if self.is_within_context_length(docs):
                               
                               return get_summary(docs,user_query)

                        else:
                                      
                                summary = get_dynamic_summary(docs,user_query)

                                return summary

                def writer(instruction:str,text_input:str):
                        """Use this tool to perform basic content writing tasks, such as summarizing previous chat history, improving responses, and more. The input should include a `query` specifying the instruction to be executed and a `text_input` containing the text on which the action is to be performed"""

                        writer_prompt = PromptTemplate(input_variables=["user_query","text_input"],
                        template= '''As a professional content writer, your task is to perform the requested operation on the provided text input according to the user's instructions. This may include tasks such as summarizing, rewriting, improving, or modifying the content as specified.

                        Text Input: The content to be processed: {text_input}

                        User Instruction: The specific task or instruction: {instruction}

                        Follow the instructions precisely to produce accurate, clear, and well-structured content. Do not include any explanations or additional messages—just generate the content as per the user's instructions.
                        ''')

                        parser = StrOutputParser()
                        chain = writer_prompt |  self.llm  | parser
                        output_text =  chain.ainvoke({"instruction":instruction,"text_input":text_input})

                        return output_text

                def page_based_content_extraction(page_number):
                        """Use this tool to extract content from a specific page. The input should be page_number, indicating the page from which the content needs to be retrieved. If the output from this tool contains <!-- image -->, you should also execute qa_with_image, as that page contains an image."""
                        
                        query = {"page_number": page_number}
                        documents = list(self.collection.find(query))
                        docs = "\n ".join(doc["text"] for doc in documents if "text" in doc)

                        try:
                               questions = self.generate_questions(docs)
                               self.st.session_state.recommended_questions = questions
                               self.recommended_questions = questions
                        except:
                               pass

                        return docs

                class ConversationAgent(TypedDict):
                        input: str
                        chat_history: list[BaseMessage]
                        agent_outcome: Union[AgentAction, AgentFinish, None]
                        intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

                answer_question_from_document_tool = StructuredTool.from_function(answer_question_from_document)
                answer_question_from_document_tool.handle_tool_error = True

                image_based_qna_tool = StructuredTool.from_function(qa_with_image)
                image_based_qna_tool.handle_tool_error = True

                user_interaction_tool = StructuredTool.from_function(user_interaction)
                user_interaction_tool.handle_tool_error = True

                summarize_document_tool = StructuredTool.from_function(summarize_document)
                summarize_document_tool.handle_tool_error = True

                writer_tool = StructuredTool.from_function(writer)
                writer_tool.handle_tool_error = True

                page_based_content_extraction_tool = StructuredTool.from_function(page_based_content_extraction)
                page_based_content_extraction_tool.handle_tool_error = True

                tools = [image_based_qna_tool,answer_question_from_document_tool, user_interaction_tool,summarize_document_tool,page_based_content_extraction_tool]

                tool_executor = ToolExecutor(tools)

                prompt_for_agent = ""

                if table_of_contents:
                       prompt_for_agent = f"Table of content of text book: {table_of_contents}"

                if self.st.session_state.document_details["total_pages"]:
                       prompt_for_agent += f"\n Total number of pages in text book: {self.st.session_state.document_details['total_pages']}"

                main_agent_prompt = ChatPromptTemplate.from_messages(
                [
                        ("system", """
                                You are an AI learning assistant with access to textbook content and specialized academic tools. Your goal is to provide accurate, contextual, and structured academic support while upholding high educational standards.  

                                Core Functions  

                                1. Content Processing  
                                - Navigate textbook content using the table of contents.  
                                - Map user queries to the most relevant textbook sections.  
                                - Maintain context across learning sessions to track topic progression.  
                                - Support incremental knowledge building for a seamless learning experience.  

                               2. Tool Integration  
                                Use the most appropriate tools based on:  
                                - Query type (conceptual explanation, problem-solving, analysis).  
                                - Required output format (text, equations, diagrams, structured responses).  
                                - Complexity level (beginner, intermediate, advanced).  
                                - Learning objectives (theoretical understanding, practical application, revision).  

                                3. Response Generation  
                                Follow a structured approach:  
                                1. Analyze the intent behind the query.  
                                2. Locate relevant content using the most suitable tools.  
                                3. Select the appropriate tool(s) for enhanced accuracy.  
                                4. Generate a structured response that is clear, concise, and easy to understand.  

                                Query Handling Strategy  

                                - If the query is clear: Proceed directly with tool selection and response generation.  
                                - If clarification is needed:  
                                - Prompt the user for specific details.  
                                - Provide an example of a well-structured query to guide refinement.  

                                ---

                                Response Structure  
                                Ensure that responses follow this structure for clarity and completeness:  
                                1. Direct Answer Concise, well-explained response tailored to the query.  
                                2. Source References  Cite textbook sections or tools used for credibility.  
                                3. Learning Reinforcement  Additional insights, real-world applications, or follow-up questions to deepen understanding.  

                                Handling Ambiguous or Incomplete Queries:  
                                - If the initial response does not fully align with the user's intent or lacks sufficient detail, rephrase and refine the query.  
                                - Execute relevant tools to fetch additional information for an improved response.  """
                                f"{prompt_for_agent}"),

                        MessagesPlaceholder("chat_history", optional=True),
                        ("human", "{input}"),
                        MessagesPlaceholder("agent_scratchpad"),
                ]
                )

                agent_runnable = create_openai_functions_agent(self.primary_llm,tools,main_agent_prompt)
                
                def run_agent(data):
                        agent_outcome = agent_runnable.invoke(data)
                        return {"agent_outcome": agent_outcome}

                def execute_tools(data):
                        agent_action = data['agent_outcome']
                        output = tool_executor.invoke(agent_action)
                        return {"intermediate_steps": [(agent_action, str(output))]}

                def should_continue(data):
                        if isinstance(data['agent_outcome'], AgentFinish):
                                return "end"
                        else:
                                return "continue"

                agentic_workflow = StateGraph(ConversationAgent)
                agentic_workflow.add_node("agent",run_agent)
                agentic_workflow.add_node("action",execute_tools)
                
                agentic_workflow.set_entry_point("agent")
                agentic_workflow.add_conditional_edges("agent",should_continue,
                        {"continue": "action","end": END})
                
                agentic_workflow.add_edge('action', 'agent')
                app = agentic_workflow.compile()

                output = app.invoke({"input": user_question, "chat_history" :chat_history, "intermediate_steps":[]})
                
                response = output['agent_outcome'].return_values['output']   

                return response, self.recommended_questions
        
        def suggest_faqs(self,user_query):
             
                suggest_faqs_prompt = PromptTemplate(input_variables=["docs","user_query"],
                template= """"You are an expert educational content creator specializing in generating high-quality academic questions and answers. Your task is to:

                1. DOCUMENT ANALYSIS:
                - Carefully analyze the provided document content in {docs}
                - Identify key concepts, important facts, and core principles
                - Consider the academic level and complexity of the material

                2. QUERY INTERPRETATION:
                - Review the user query in {user_query} for specific requirements
                - If no specific requirements are given, focus on:
                - Core concepts from the document
                - Different levels of cognitive understanding (recall, comprehension, application)
                - Key learning objectives implicit in the content

                3. QUESTION GENERATION GUIDELINES:
                - Create questions that:
                - Are clear and unambiguous
                - Use appropriate academic language
                - Test different cognitive levels (Bloom's Taxonomy)
                - Have exactly four options per question
                - Include plausible distractors based on common misconceptions
                - Avoid negative phrasing (e.g., "Which of the following is NOT...")
                - Are grammatically consistent between question and options

                4. QUALITY ASSURANCE:
                - Ensure each question:
                - Has exactly one correct answer
                - Has unique options (no duplicates)
                - Is relevant to the document content
                - Is factually accurate
                - Has clear, unambiguous wording

                5. OUTPUT FORMAT:
                Generate the output in the following JSON structure:
                ```json
                [
                [
                "question": "Clear, well-formed question text",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer": "Exact match to one of the options"
                ],
                ...
                ]
                ```

                6. SPECIAL CONSIDERATIONS:
                - For mathematical content: Include units where applicable
                - For scientific content: Use proper scientific notation and terminology
                - For historical content: Ensure dates and events are precise
                - For conceptual content: Focus on understanding rather than mere memorization

                7. DIFFICULTY DISTRIBUTION:
                Unless specified otherwise in the user query, create a mix of:
                - 25% Basic recall questions
                - 50% Understanding/application questions
                - 25% Analysis/evaluation questions

                8. ERROR HANDLING:
                If the document content is:
                - Empty: Return an error message requesting content
                - Unclear: Generate questions about the clearly understood portions only
                - Too technical: Adapt the language to appropriate academic level

                Remember to maintain academic integrity and educational value in all generated questions. Each question should contribute to the learner's understanding of the subject matter.
                
                output should be an valid json in list of dict format""")


                docs = mongodb_hybrid_search(query=user_query,top_k=4,collection=self.collection)
                faqs_chain = suggest_faqs_prompt | self.primary_llm | JsonOutputParser()
                faqs = faqs_chain.invoke({"docs":docs,"user_query":user_query})
                return faqs


        

