# %%
# %env LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# %env LLM_API_KEY=替换为自己的key

# %%
# %%capture --no-stderr
# !pip install -U langchain langchain_community langchain_openai pypdf sentence_transformers chromadb shutil openpyxl

# %%
# import langchain, langchain_community, pypdf, sentence_transformers, chromadb, langchain_core

# for module in (langchain, langchain_core, langchain_community, pypdf, sentence_transformers, chromadb):
#     print(f"{module.__name__:<30}{module.__version__}")

# %%
import os
import pandas as pd

# %%
expr_version = 'retrieval_v3_rag_fusion'

preprocess_output_dir = r""
expr_dir = os.path.join(os.path.pardir, 'experiments', expr_version)

# %% [markdown]
# # 读取文档

# %%
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"")
documents = loader.load()

qa_df = pd.read_excel(os.path.join(preprocess_output_dir, 'question_answer.xlsx'))

# %% [markdown]
# # 文档切分

# %%
from uuid import uuid4
import os
import pickle

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents, filepath, chunk_size=400, chunk_overlap=40, seperators=['\n\n\n', '\n\n'], force_split=False):
    if os.path.exists(filepath) and not force_split:
        print('found cache, restoring...')
        return pickle.load(open(filepath, 'rb'))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=seperators
    )
    split_docs = splitter.split_documents(documents)
    for chunk in split_docs:
        chunk.metadata['uuid'] = str(uuid4())

    pickle.dump(split_docs, open(filepath, 'wb'))

    return split_docs

# %%
splitted_docs = split_docs(documents, os.path.join(preprocess_output_dir, 'split_docs.pkl'), chunk_size=500, chunk_overlap=50)

# %% [markdown]
# # 检索

# %%
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')

def get_embeddings(model_path):
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_path,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True},
        # show_progress=True
        query_instruction='为这个句子生成表示以用于检索相关文章：'
    )
    return embeddings

# %%
import shutil
from langchain_community.vectorstores import Chroma

model_path = 'BAAI/bge-large-zh-v1.5'

persist_directory = os.path.join(expr_dir, 'chroma')
shutil.rmtree(persist_directory, ignore_errors=True)

embeddings = get_embeddings(model_path)
vector_db = Chroma.from_documents(
    splitted_docs,
    embedding=embeddings,
    persist_directory=persist_directory
)

# %%
test_df = qa_df[(qa_df['dataset'] == 'test') & (qa_df['qa_type'] == 'detailed')]

# %% [markdown]
# ## 不使用RAG Fusion

# %%
def get_emb_retriever(k):
    return vector_db.as_retriever(search_kwargs={'k': k})

# %%
from tqdm.auto import tqdm

def get_hit_stat_df(get_retriever_fn, top_k_arr=list(range(1, 9))):
    hit_stat_data = []
    pbar = tqdm(total=len(top_k_arr) * len(test_df))
    for k in top_k_arr:
        pbar.set_description(f'k={k}')
        retriever = get_retriever_fn(k)
        
        for idx, row in test_df.iterrows():
            question = row['question']
            true_uuid = row['uuid']
            
            chunks = retriever.invoke(question)[:k]
            retrieved_uuids = [doc.metadata['uuid'] for doc in chunks]

            hit_stat_data.append({
                'question': question,
                'top_k': k,
                'hit': int(true_uuid in retrieved_uuids),
                'retrieved_chunks': len(chunks)
            })
            pbar.update(1)
    hit_stat_df = pd.DataFrame(hit_stat_data)
    return hit_stat_df

# %%
orig_query_hit_stat_df = get_hit_stat_df(get_emb_retriever)
orig_query_hit_stat_df['rag_fusion'] = 'w/o'

# %% [markdown]
# ## 使用RAG Fusion

# %%
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
import re

llm = ChatOllama(base_url='http://localhost:11434', model='qwen2:7b-instruct')

prompt = PromptTemplate(
    input_variables=['question', 'n_sim_query'],
    template = """你是一个AI语言模型助手。你的任务是基于给定的原始问题，再生成出来最相似的{n_sim_query}个不同的版本。\
你的目标是通过生成用户问题不同视角的版本，帮助用户克服基于距离做相似性查找的局限性。\
使用换行符来提供这些不同的问题，使用换行符来切分不同的问题，不要包含数字序号，仅返回结果即可，不要添加任何其他描述性文本。
原始问题：{question}
"""
)

generate_queries_chain = (
    prompt
    | llm
    # 有时候模型不遵循指令，把前面的序号去掉
    # | (lambda x: re.sub(r'\d+\.\s', '', x.content))
    # 有时候模型不遵循指令，把前面的序号、- 去掉
    | (lambda x: [
        re.sub(r'(^\-\s+)|(^\d+\.\s)', '', item.strip()) 
        for item in x.content.split('\n') if item.strip() != ''
    ])
)

# %%
from langchain.load import dumps, loads

def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        # 此处有一个隐含的假设：返回的docs是按相似度排好序的
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

# %%
def get_rag_fusion_chain(top_k, n_sim_query=3, trunc=False):
    """
    获取RAG Fusion Chain
    :param top_k: 每个相似问检索的片段数量
    :param trunc: 最终排序后的结果是否要截断为top_k
    """
    chain = (
        prompt.partial(n_sim_query=n_sim_query)
        | llm
        # 有时候模型不遵循指令，把前面的序号去掉
        # | (lambda x: re.sub(r'\d+\.\s', '', x.content))
        # 有时候模型不遵循指令，把前面的序号、- 去掉
        | (lambda x: [
            re.sub(r'(^\-\s+)|(^\d+\.\s)', '', item.strip()) 
            for item in x.content.split('\n') if item.strip() != ''])
        | (lambda x: [vector_db.similarity_search(q, k=top_k) for q in x])
        | reciprocal_rank_fusion
        | (lambda docs: docs[:top_k] if trunc else docs)
    )
    return chain

# %% [markdown]
# 首先先生成相似问题，这个跟Multi Query是类似的

# %%
generate_queries_chain.invoke({'question': '报告的发布机构是什么？', 'n_sim_query': 3})

# %% [markdown]
# 查看一下中间过程，这个步骤，对于每一个产生的相似问题，检索5个片段

# %%
retrieved_docs = (
    generate_queries_chain
    | (lambda x: [vector_db.similarity_search(q, k=5) for q in x])
).invoke({'question': '报告的发布机构是什么？', 'n_sim_query': 3})

# %%
len(retrieved_docs)

# %%
len(retrieved_docs[0])

# %%
len(get_rag_fusion_chain(3).invoke({'question': '报告的发布机构是什么？'}))

# %% [markdown]
# 如果确认rag_fusion_chain没有问题，可以直接设置trunc为True，确保召回数量符合设定值

# %%
get_rag_fusion_chain(2, trunc=True).invoke({'question': '报告的发布机构是什么？'})

# %% [markdown]
# 为了方便调参，我们创建一个新的函数，使用下面这个函数

# %%
def retrieve_with_rrf(llm, query, top_k=4, n_sim_query=3, include_original=True, trunc=True):
    """
    使用RRF检索
    :param llm: 用于生成相似问题的LLM
    :param query: 需要检索的问题
    :param top_k: 每个问题检索几个知识片段
    :param n_sim_query: 每个query生成几个相似问题
    :param include_original: 检索知识片段时，是否包含原始问题
    :param trunc: 是否将最终检索结果，截断为top_k
    """
    chain = (
        prompt.partial(n_sim_query=n_sim_query)
        | llm
        # 有时候模型不遵循指令，把前面的序号、- 去掉
        | (lambda x: [
            re.sub(r'(^\-\s+)|(^\d+\.\s)', '', item.strip()) 
            for item in x.content.split('\n') if item.strip() != ''])
        | (lambda x: [vector_db.similarity_search(q, k=top_k) for q in ([query] if include_original else []) + x[:n_sim_query]])
        | reciprocal_rank_fusion
        | (lambda docs: docs[:top_k] if trunc else docs)
    )
    return chain.invoke(query)

# %%
def get_rag_fusion_hit_stat_df(query_gen_llm, top_k_arr=list(range(1, 9))):
    hit_stat_data = []
    pbar = tqdm(total=len(top_k_arr) * len(test_df))
    for k in top_k_arr:
        pbar.set_description(f'k={k}')
        rag_fusion_chain = get_rag_fusion_chain(k, trunc=True)
        for idx, row in test_df.iterrows():
            question = row['question']
            true_uuid = row['uuid']
            # chunks = rag_fusion_chain.invoke({'question': question})
            chunks = retrieve_with_rrf(query_gen_llm, question, top_k=k)
            assert len(chunks) <= k
            
            retrieved_uuids = [doc.metadata['uuid'] for doc, score in chunks]

            hit_stat_data.append({
                'question': question,
                'top_k': k,
                'hit': int(true_uuid in retrieved_uuids),
                'retrieved_chunks': len(chunks)
            })
            pbar.update(1)
    hit_stat_df = pd.DataFrame(hit_stat_data)
    return hit_stat_df

# %%
from langchain_openai import ChatOpenAI

qwen2_14b_llm = ChatOpenAI(
    base_url=os.environ['LLM_BASE_URL'],
    api_key=os.environ['LLM_API_KEY'],
    model='qwen2-57b-a14b-instruct'
)

rag_fusion_hit_stat_dfs = []
for llm_name, llm_model in zip(['ollama-qwen2-7b-instruct', 'qwen2-57b-a14b-instruct'], [llm, qwen2_14b_llm]):
    rag_fusion_hit_stat_df = get_rag_fusion_hit_stat_df(llm_model)
    rag_fusion_hit_stat_df['rag_fusion'] = f'w/ {llm_name}'
    rag_fusion_hit_stat_dfs.append(rag_fusion_hit_stat_df)

# %%
hit_stat_df = pd.concat([orig_query_hit_stat_df] + rag_fusion_hit_stat_dfs)

# %%
hit_stat_df.groupby(['rag_fusion', 'top_k'])['hit'].mean().reset_index().rename(columns={'hit': 'hit_rate'})

# %%
import seaborn as sns

sns.barplot(x='top_k', y='hit', hue='rag_fusion', data=hit_stat_df, errorbar=None)

# %% [markdown]
# # 预测

# %%
from langchain.llms import Ollama

ollama_llm = Ollama(
    model='qwen2:7b-instruct',
    base_url='http://localhost:11434'
)

# %%
ollama_llm.invoke('你是谁')

# %%
def rag(query_gen_llm, question, n_chunks=3):
    prompt_tmpl = """
你是一个金融分析师，擅长根据所获取的信息片段，对问题进行分析和推理。
你的任务是根据所获取的信息片段（<<<<context>>><<<</context>>>之间的内容）回答问题。
回答保持简洁，不必重复问题，不要添加描述性解释和与答案无关的任何内容。
已知信息：
<<<<context>>>
{{knowledge}}
<<<</context>>>

问题：{{question}}
请回答：
""".strip()

    # rag_fusion_chain = get_rag_fusion_chain(n_chunks, trunc=True)
    # chunks = rag_fusion_chain.invoke({'question': question})
    
    chunks = retrieve_with_rrf(query_gen_llm, question, top_k=n_chunks)
    prompt = prompt_tmpl.replace('{{knowledge}}', '\n\n'.join([pair[0].page_content for pair in chunks])).replace('{{question}}', question)

    return ollama_llm(prompt), chunks

# %%
print(rag(llm, '2023年10月美国ISM制造业PMI指数较上月有何变化？')[0])

# %%
prediction_df = qa_df[qa_df['dataset'] == 'test'][['uuid', 'question', 'qa_type', 'answer']].rename(columns={'answer': 'ref_answer'})

def predict(query_gen_llm, prediction_df, n_chunks):
    prediction_df = prediction_df.copy()
    answer_dict = {}

    for idx, row in tqdm(prediction_df.iterrows(), total=len(prediction_df)):
        uuid = row['uuid']
        question = row['question']
        answer, chunks = rag(query_gen_llm, question, n_chunks=n_chunks)
        assert len(chunks) <= n_chunks
        answer_dict[question] = {
            'uuid': uuid,
            'ref_answer': row['ref_answer'],
            'gen_answer': answer,
            'chunks': chunks
        }
    prediction_df.loc[:, 'gen_answer'] = prediction_df['question'].apply(lambda q: answer_dict[q]['gen_answer'])
    prediction_df.loc[:, 'chunks'] = prediction_df['question'].apply(lambda q: answer_dict[q]['chunks'])

    return prediction_df

# %%
n_chunks = 3

pred_df_dict = {}
for llm_name, llm_model in zip(['ollama-qwen2-7b-instruct', 'qwen2-57b-a14b-instruct'], [llm, qwen2_14b_llm]):
    pred_df_dict[llm_name] = predict(llm_model, prediction_df, n_chunks=n_chunks)

# %% [markdown]
# # 评估

# %%
from langchain_openai import ChatOpenAI
import time

judge_llm = ChatOpenAI(
    api_key=os.environ['LLM_API_KEY'],
    base_url=os.environ['LLM_BASE_URL'],
    model_name='qwen2-72b-instruct',
    temperature=0
)

def evaluate(prediction_df):
    """
    对预测结果进行打分
    :param prediction_df: 预测结果，需要包含问题，参考答案，生成的答案，列名分别为question, ref_answer, gen_answer
    :return 打分模型原始返回结果
    """
    prompt_tmpl = """
你是一个经济学博士，现在我有一系列问题，有一个助手已经对这些问题进行了回答，你需要参照参考答案，评价这个助手的回答是否正确，仅回复“是”或“否”即可，不要带其他描述性内容或无关信息。
问题：
<question>
{{question}}
</question>

参考答案：
<ref_answer>
{{ref_answer}}
</ref_answer>

助手回答：
<gen_answer>
{{gen_answer}}
</gen_answer>
请评价：
    """
    results = []

    for _, row in tqdm(prediction_df.iterrows(), total=len(prediction_df)):
        question = row['question']
        ref_answer = row['ref_answer']
        gen_answer = row['gen_answer']

        prompt = prompt_tmpl.replace('{{question}}', question).replace('{{ref_answer}}', str(ref_answer)).replace('{{gen_answer}}', gen_answer).strip()
        result = judge_llm.invoke(prompt).content
        results.append(result)

        time.sleep(1)
    return results

# %%
for model_name, pred_df in pred_df_dict.items():
    pred_df['raw_score'] = evaluate(pred_df)
    print(f"{model_name}: {pred_df['raw_score'].unique()}")
    pred_df.loc[:, 'score'] = pred_df['raw_score'].replace({'是': 1, '否': 0})

# %%
pred_df_dict['ollama-qwen2-7b-instruct']['score'].mean()

# %%
pred_df_dict['qwen2-57b-a14b-instruct']['score'].mean()

# %%
for model_name, pred_df in pred_df_dict.items():
    pred_df.to_excel(os.path.join(expr_dir, f'prediction_{model_name}.xlsx'), index=False)

# %%



