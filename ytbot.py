import gradio as gr
import re
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from ibm_watsonx_ai.foundation_models.utils import get_embedding_model_specs
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


def get_video_id(url):
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None


url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
video_id = get_video_id(url)
print(video_id)


def get_transcript(url):
    video_id = get_video_id(url)

    if not video_id:
        return None

    ytt = YouTubeTranscriptApi()
    transcript_list = ytt.list(video_id)

    trans = None

    for t in transcript_list:
        if t.language_code == "en":
            trans = t.fetch()
            break

    if trans is None:
        for t in transcript_list:
            if t.is_generated:
                trans = t.fetch()
                break

    return trans


def process(trans):
    if not trans:
        return ""

    s = ""

    for i in trans:
        s = s + f"Text:{i.text} start:{i.start}\n"

    return s


def chunk_transcript(trans, chunk_size=200, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    r = text_splitter.split_text(trans)

    return r


def set_up_credentials():
    model_id = "ibm/granite-8b-code-instruct"

    credentials = Credentials(
        url="https://us-south.ml.cloud.ibm.com"
    )

    client = APIClient(credentials)

    project_id = "skills-network"

    return model_id, credentials, client, project_id


def define_params():
    return {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MAX_NEW_TOKENS: 900
    }


def instialize_model(model_id, credentials, project_id, parameters):
    return WatsonxLLM(
        model_id=model_id,
        url=credentials.get("url"),
        project_id=project_id,
        params=parameters
    )


def setup_embedding_model(credentials, project_id):
    return WatsonxEmbeddings(
        model_id="ibm/slate-30m-english-rtrvr-v2",
        url=credentials.get("url"),
        project_id=project_id
    )


def create_faiss_index(chunks, embedding_model):
    return FAISS.from_texts(chunks, embedding_model)


def perform_similarity_function(faiss_index, query, k=3):
    results = faiss_index.similarity_search(query, k)
    return results


def create_summary_prompt():
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps in your summary.
    3. Focus on the spoken content (Text) of the video.

    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Please summarize the following YouTube video transcript:

    {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    prompt = PromptTemplate(
        input_variables=["transcript"],
        template=template
    )

    return prompt


def create_summary_chain(llm, prompt, verbose=True):
    return LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=verbose
    )


def retrieve(query, faiss_index, k=3):
    relevant_context = faiss_index.similarity_search(
        query,
        k=k
    )

    return relevant_context


def create_qa_prompt_template():
    qa_template = """
    You are an expert assistant providing detailed answers based on the following video content.

    Relevant Video Context: {context}

    Based on the above context, please answer the following question:
    Question: {question}
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )

    return prompt


def create_qa_chain(llm, prompt_template, verbose=True):
    return LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=verbose
    )


def generate_answer(question, faiss_index, qa_chain, k=7):
    relevant_context = retrieve(
        question,
        faiss_index,
        k=k
    )

    answer = qa_chain.predict(
        context=relevant_context,
        question=question
    )

    return answer


fetched = None
processed = ""


def summarize_video(video_url):
    global fetched, processed

    if video_url:
        fetched = get_transcript(video_url)
        processed = process(fetched)
    else:
        return "provide valid youtube link"

    if processed:
        model_id, credentials, client, project_id = set_up_credentials()

        llm = instialize_model(
            model_id,
            credentials,
            project_id,
            define_params()
        )

        summary_prompt = create_summary_prompt()

        summary_chain = create_summary_chain(
            llm,
            summary_prompt
        )

        summary = summary_chain.run(
            {"transcript": processed}
        )

        return summary

    else:
        return "No transcript available"


def answer_question(video_url, user_question):
    global fetched, processed

    if not processed:
        if video_url:
            fetched = get_transcript(video_url)
            processed = process(fetched)
        else:
            return "please provide a valid youtube link"

    if processed and user_question:
        chunks = chunk_transcript(processed)

        model_id, credentials, client, project_id = set_up_credentials()

        llm = instialize_model(
            model_id,
            credentials,
            project_id,
            define_params()
        )

        embedding_model = setup_embedding_model(
            credentials,
            project_id
        )

        faiss_index = create_faiss_index(
            chunks,
            embedding_model
        )

        qa_prompt = create_qa_prompt_template()

        qa_chain = create_qa_chain(
            llm,
            qa_prompt
        )

        answer = generate_answer(
            user_question,
            faiss_index,
            qa_chain
        )

        return answer

    else:
        return "please provide a valid question"


with gr.Blocks() as interface:

    video_url = gr.Textbox(
        label="YouTube Video URL",
        placeholder="Enter the YouTube Video URL"
    )

    summary_output = gr.Textbox(
        label="Video Summary",
        lines=5
    )

    question_input = gr.Textbox(
        label="Ask a Question About the Video",
        placeholder="Ask your question"
    )

    answer_output = gr.Textbox(
        label="Answer to Your Question",
        lines=5
    )

    summarize_btn = gr.Button("Summarize Video")

    question_btn = gr.Button("Ask a Question")

    transcript_status = gr.Textbox(
        label="Transcript Status",
        interactive=False
    )

    summarize_btn.click(
        summarize_video,
        inputs=video_url,
        outputs=summary_output
    )

    question_btn.click(
        answer_question,
        inputs=[video_url, question_input],
        outputs=answer_output
    )


interface.launch(
    server_name="0.0.0.0",
    server_port=7860
)