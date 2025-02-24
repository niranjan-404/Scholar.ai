import streamlit as st
from processing.processing import DocumentProcessing
from datetime import datetime
import pytz
import os
from langchain_core.messages import AIMessage, HumanMessage

UPLOAD_FOLDER = "uploads"

def get_greeting(hour):
    if hour < 12:
        return "🌅 Good Morning, Student!"
    elif hour < 17:
        return "☀️ Good Afternoon, Student!"
    else:
        return "🌙 Good Evening, Student!"

def main():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    st.markdown('# 📚 Scholar.ai', unsafe_allow_html=True)

    st.markdown('<p class="subtitle">Your Intelligent Learning Companion</p>', unsafe_allow_html=True)

    ist = pytz.timezone('Asia/Kolkata')
    hour = datetime.now(ist).time().hour
    greeting = get_greeting(hour)
    st.markdown(f'<p class="welcome-message"> {greeting}, Scholar!</p>', unsafe_allow_html=True)

    
    st.markdown(""" **Hey there, future achiever!** 👋  
        
    I'm your ultimate **study buddy**, here to make learning easier, faster, and way more fun! 🚀   
    """)

    st.markdown("""
        
        Just upload your study material—whether it's a textbook, notes, or even a research paper—and I'll help you: """)
    
    st.markdown("""
        - **📖 Simplify complex concepts** into bite-sized, easy-to-understand ideas  
        - **❓ Answer your burning questions** 
        - **📝 Create crisp summaries** so you can focus on what matters  
        - **🎯 Generate practice questions** to test your knowledge and ace your exams  
        - **🖼️ Explain images, diagrams, or discussions** on any page  
        - **📌 Build FAQs** for quick revision  
        - **📓 Take notes** on specific topics or page ranges—just tell me what you need!  

        Think of me as your **24/7 study partner**—here to make your study sessions smarter, not harder. Ready to crush your goals? Let's get started! 🚀 """)
    
    
    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    
    uploaded_file = st.file_uploader(
        "📎 Drop your study material here",
        type=["pdf"],
        accept_multiple_files=False
    )
    
    if uploaded_file:
            st.success(f"🎉 Thank you for sharing {uploaded_file.name}! I'll review it and then we can have a discussion about it.😊")
            
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)

            file_details = {
                "📄 Document Name": uploaded_file.name,
                "📊 Size": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.json(file_details)
            st.session_state.document_details = file_details

            with st.spinner("🔍 Reading through your document... This might take a moment!"):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                documentProcessor = DocumentProcessing(file_path,st)

                total_pages = documentProcessor.get_pdf_page_count()

                st.session_state.document_details["total_pages"] = total_pages

                if not documentProcessor.collection.find_one({"source":uploaded_file.name}):
                
                    mark_down_docs , data_for_model = documentProcessor.load_markdown_document()

                    result = documentProcessor.extract_table_of_contents(text_input=data_for_model)

                    table_of_contents = result["table_of_contents"]

                    documentProcessor.page_number_correction = result["page_numbering_correction"]

                    documentProcessor.table_of_contents = table_of_contents
        
                    documentProcessor.create_vectorStore(docs=mark_down_docs)

                    documentProcessor.collection.insert_one({"field":"table_of_contents","data":result["table_of_contents"],"page_numbering_correction":result["page_numbering_correction"]})

                
                else:

                    cursor = documentProcessor.collection.find_one({"field":"table_of_contents"})

                    if cursor:

                        table_of_contents = cursor.get("data")

                        documentProcessor.table_of_contents = table_of_contents

                        documentProcessor.page_number_correction = cursor.get("page_numbering_correction")


            st.session_state["recommended_questions"] = []

            st.success("🎉 Your document is ready for learning!")

            tabs = st.tabs(["🤔 Ask Me Anything", "📝 Get Quick Notes", "🎯 Practice Questions" ])
            
            def clear_input():
                st.session_state.user_input = ""

            st.markdown("""
                <style>
                .stButton > button {
                    margin-top: 0px;
                    padding-top: 0.70rem;
                    padding-bottom: 0.70rem;
                    height: 38px;  
                }
                div.row-widget.stButton {
                    margin-top: 0px;
                    padding-top: 0px;
                }
                div.row-widget.stTextInput > div {
                    padding-top: 0px;
                    padding-bottom: 0px;
                }
                /* Add styling for the button container */
                div.stButton > button:first-child {
                    margin-top: 3px;  
                }
                </style>
            """, unsafe_allow_html=True)

            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            if 'recommended_questions' not in st.session_state:
                st.session_state.recommended_questions = []
            if 'user_input' not in st.session_state:
                st.session_state.user_input = None
            if 'selected_question' not in st.session_state:
                st.session_state.selected_question = None

            with tabs[0]:
                st.markdown("### 🤔 What's on your mind?")
                st.markdown("Ask me anything about your document - no question is too simple!")
                
                for message in st.session_state.chat_history:
                    if isinstance(message, HumanMessage):
                        with st.chat_message("user", avatar="🎓"):
                            st.write(message.content)
                    elif isinstance(message, AIMessage):
                        with st.chat_message("assistant", avatar="🤖"):
                            st.write(message.content)


                if st.session_state.recommended_questions:
                    for question in st.session_state.recommended_questions:
                        if st.button(question, key=f"{question}"):
                            st.session_state.selected_question = question

                            
                col1, col2 = st.columns([7, 1], gap="small")
                with col1:
                    user_question = st.text_input(
                        "Type your question here...",
                        key="text_input",
                        value=st.session_state.selected_question,
                        label_visibility="collapsed"
                    )
                    
                with col2:
                    submit = st.button("Submit", on_click=clear_input)
                    
                if submit and user_question:
                    with st.spinner("🧠 Thinking..."):
                        response, recommended_questions = documentProcessor.chat_with_document(
                            st.session_state.chat_history,
                            user_question
                        )
                        
                        st.session_state.chat_history.append(HumanMessage(content=user_question))
                        st.session_state.chat_history.append(AIMessage(content=response))
                        
                        st.session_state.recommended_questions = recommended_questions

                        st.rerun()

            
            with tabs[1]:
                st.markdown("### 📝 Quick Study Notes")
                st.markdown("Let me help you break this down into bite-sized pieces!")
                
                notes_tabs = st.tabs(["📑 Selected Pages","📌 On specific topic"])
                
                if 'topic_level_notes' not in st.session_state:
                    st.session_state["topic_level_notes"] = []
                if 'pages_level_notes' not in st.session_state:
                    st.session_state["pages_level_notes"] = []
                    
                with notes_tabs[0]:
                    st.markdown("📑 Choose your pages:")
                    col1, col2 = st.columns(2)
                    with col1:
                        start_page = st.number_input("Start from page", 1, total_pages, 1)
                    with col2:
                        end_page = st.number_input("End at page", start_page+1, total_pages)
                    
                    if st.button("📝 Create Notes for Selected Pages") and start_page and end_page:
                        with st.spinner("🎨 Creating notes for your selected pages..."):
                            notes = documentProcessor.generate_notes(start_page,end_page)
                            st.write("✨ Here's your Notes from selected pages:")
                            st.markdown(notes)
                            st.session_state["pages_level_notes"].extend([
                                HumanMessage(content=f"Create notes from page number {start_page} to {end_page}"),
                                AIMessage(content=notes)
                            ])

                with notes_tabs[1]:

                    topic = st.text_input("Type your preferred topic here...")
                    
                    if st.button("📝 Create Notes for the selected topic"):
                        with st.spinner("🎨 Creating notes for your selected topic..."):
                            notes = documentProcessor.generate_notes(topic=topic)
                            st.write("✨ Here's your Notes:")
                            st.markdown(notes)
                            st.session_state["topic_level_notes"].extend([
                                HumanMessage(content=f"Create notes for `{topic}`"),
                                AIMessage(content=notes)
                            ])
            
            with tabs[2]:
                st.markdown("### 🎯 Practice Questions")
                st.markdown("Let's test your understanding with some practice questions!")

                user_question = st.text_input("Type your topic here on which questions are to be created...")
                
                marks_per_question = 1
                if st.button("🎲 Generate Study Questions"):
                    with st.spinner("🧩 Creating engaging questions..."):
                        questions = documentProcessor.suggest_faqs(user_question)
                        st.session_state.questions = questions
                        st.session_state.submitted = False
                        st.session_state.user_answers = [None] * len(questions)
                        st.session_state.total_possible_marks = len(questions) * marks_per_question

                if "questions" in st.session_state:
                    st.title("Quiz")
                    st.markdown(f"Total possible marks: {st.session_state.total_possible_marks}")
                    
                    for idx, question_data in enumerate(st.session_state.questions):
                        st.subheader(f"Q{idx + 1}: {question_data['question']} ({marks_per_question} mark)")
                        
                        selected_option = st.radio(
                            "Select an option:",
                            question_data["options"],
                            index=None if st.session_state.user_answers[idx] is None 
                                else question_data["options"].index(st.session_state.user_answers[idx]),
                            key=f"question_{idx}"
                        )
                        
                        st.session_state.user_answers[idx] = selected_option
                        
                        if st.session_state.submitted:
                            if selected_option == question_data["correct_answer"]:
                                st.success(f"✅ Correct! (+{marks_per_question} mark)")
                            else:
                                st.error(f"❌ Wrong! Correct answer: {question_data['correct_answer']} (0 marks)")
                        
                        st.markdown("---") 
                    
                    if st.button("Submit Quiz"):
                        if None in st.session_state.user_answers:
                            st.warning("Please answer all questions before submitting!")
                        else:
                            st.session_state.submitted = True

                            correct_answers = sum(
                                1 for user_ans, q in zip(st.session_state.user_answers, st.session_state.questions)
                                if user_ans == q["correct_answer"]
                            )
                            total_marks = correct_answers * marks_per_question
                            
                            st.markdown("### Quiz Results")
                            st.markdown(f"Number of correct answers: {correct_answers}/{len(st.session_state.questions)}")
                            st.markdown(f"Total marks obtained: {total_marks}/{st.session_state.total_possible_marks}")
                            
                            percentage = (total_marks / st.session_state.total_possible_marks) * 100
                            st.markdown(f"Percentage: {percentage:.1f}%")
                            
                            if percentage >= 90:
                                st.success("🌟 Excellent performance!")
                            elif percentage >= 70:
                                st.success("👍 Good job!")
                            elif percentage >= 50:
                                st.warning("🎯 Pass! But there's room for improvement.")
                            else:
                                st.error("📚 Need more practice!")
                    
                    if st.session_state.submitted:
                        if st.button("Try Again"):
                            st.session_state.submitted = False
                            st.session_state.user_answers = [None] * len(st.session_state.questions)
                            st.rerun()

if __name__ == "__main__":
    main()















