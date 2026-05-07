# 在半个月的时间内，完成大模型接入、环境配置、ollama本地部署，以及OpenAI库基础使用、提示词工程优化的学习，完成LangChain框架环境部署，RAG开发部分知识的学习。具体如下：

一、大模型接入与环境配置

选取千问API，去阿里的大模型百炼服务平台获取API KEY，并在高级系统设置中的环境变量添加变量名为DASHSCOPE_API_KEY的环境变量以保护API KEY

from openai import OpenAI

#获取客户端对象
client = OpenAI(
 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
二、ollama本地部署

官网下载，安装，再使用代码调用本地模型。


安装后界面
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
)
completion = client.chat.completions.create(
    model="qwen3:4b",
三、OpenAI库基础使用

OpenAI库是OpenAI官方推出的Python SDK，核心作用是让开发者能简单、高效地调用 OpenAI 的各类 API（如 GPT 聊天、DALL·E 绘图、语音转文字等），无需手动处理 HTTP 请求、身份验证等底层细节。现如今许多模型服务商（如阿里云百炼平台）均兼容OpenAI SDK的调用。

1.基础使用

使用流程：获取客户端对象->调用模型->处理结果

获取客户端对象：

client = OpenAI(
 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
调用模型：client.chat.completions.create()创建ChatCompletetion对象，主要参数为model和messages（列表，包含多个字典信息，每个字典包含两个key，分别role和content）。

response=client.chat.completions.create(
 model="qwen-max",
 messages=[
        {"role":"system", "content":"你是一个python编程专家，且不说废话。"},
        {"role":"assistant", "content":"我是一个不说废话的python编程专家，你要问什么？"},
        {"role":"user", "content":"用python代码输出1-10的数字"}
    ],
)
处理结果：用print(response.choices[0].message.content)获取结果

print(response.choices[0].message.content)
2.流式输出

两步走，第一步，在client.chat.completions.create()调用模型时设置参数stream=True；第二步，for循环response对象，并在循环内输出内容。

#调用模型
messages = [{"role": "user", "content": "你能做什么"}]
completion = client.chat.completions.create( # 创建chatcompletetion对象
    model="qwen3-max",  # 您可以按需更换为其它深度思考模型
    messages=messages,
    extra_body={"enable_thinking": True},
    stream=True
)
is_answering = False  # 是否进入回复阶段
print("\n" + "=" * 20 + "思考过程" + "=" * 20)
for chunk in completion:
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
        if not is_answering:
            print(delta.reasoning_content, end="", flush=True)
    if hasattr(delta, "content") and delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + "完整回复" + "=" * 20)
            is_answering = True
        print(delta.content, end="", flush=True)
3.附带历史消息调用模型

在messages中将历史消息填入，让模型知晓对话的上下文，更好地回答

response = client.chat.completions.create(
    model="qwen3-max",
    messages=[
        {"role": "system", "content": "你是AI助理，回答很简洁"},
        {"role": "user", "content": "小明有2条宠物狗"},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": "小红有3只宠物猫"},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": "总共有几个宠物？"}
    ],
    stream=True    # 开启了流式输出的功能
)

for chunk in response:
    print(
        chunk.choices[0].delta.content,
        end=" ",     #每段之间以空格分开
        flush=True   #立刻刷新缓冲区
    )
四、提示词工程优化

1.写提示词时的技巧：详细描述、让模型充当某个角色、使用分隔符、对任务提示步骤、提供例子

2.Zero-shot Learning是指在训练阶段不存在与测试阶段完全相同的类别，基于已训练的能力，不提供任何示例，仅通过语言去描述任务的要求、目标和约束，让模型直接生成结果，是“语言定义任务”。Few-shot Learning是指当模型在学习了一定类别的大量数据后，对于新的类别，只需要少量的样本就能快速学习，是“用示例定义任务”。

3.文本分类prompt设计

在文本分类的prompt设计中，主要考虑两点，一，向模型解释什么叫作「文本分类任务」；二， 让模型按照我们指定的格式输出，即借用 FewShot 的方式，给模型展示一些正确的例子：

User: “今日，股市经历了一轮震荡，受到宏观经济数据和全球贸易紧张局势的影响。投资者密切关注美联储可能的政策调整，以适应市场的不确定性。”是[新闻报道]、[公司公告]、[财务公告]、[分析师报告]里的什么类别？

Bot: 新闻报道

from openai import OpenAI

client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 示例类别
examples_data = {
    "新闻报道": "今日，股市经历了一轮震荡......",
    "财务报告": "本公司年度财务报告显示......",
    "公司公告": "本公司高兴地宣布成功完成最新一轮并购交易......",
    "分析师报告": "最新的行业分析报告指出......"
}

# 待分类文本
questions = [
    "今日，央行发布公告宣布降低利率......",
    "ABC公司今日发布公告称......",
    "公司资产负债表显示......",
    "最新的分析报告指出......",
    "小明喜欢小新帅"
]

# 构造messages
messages = [
    {"role": "system","content": "你是金融专家，将文本分类为['新闻报道','财务报告','公司公告','分析师报告']，不清楚的分类为'不清楚类别'"}
]
for key, value in examples_data.items():
    messages.append({"role": "user", "content": value})
    messages.append({"role": "assistant", "content": key})

# 循环调用模型，对每个问题单独发一次请求
for q in questions:
    response = client.chat.completions.create(
        model="qwen3-max",
        messages=messages + [{"role": "user", "content": f"按照示例，回答这段文本的分类类别：{q}"}]
    )

    print(response.choices[0].message.content)
4.信息抽取prompt设计

在文本分类的prompt设计中，主要考虑两点，一，向模型解释什么叫作「信息抽取任务」；二， 让模型按照我们指定的格式（json）输出，即借用 FewShot 的方式，给模型展示一些正确的例子：

User:
2023-01-10，股市震荡。股票古哥-D[EOOE]美股今日开盘价100美元，一度飙升至105美元，随后回落至98美元，最终以102美元收盘，成交量达到520000。
提取上述句子中「金融」（日期；股票名称；开盘价；收盘价；成交量）类型的实体，并按照JSON格式输出，上述句子中没有的信息用「原文中未提及」来表示，多值之间用/分隔。

Bot:
{"日期": ["2023-01-10"], "股票名称": ["古哥-D[EOOE]美股"], "开盘价": ["100美元"], "收盘价": ["102美元"], "成交量": ["520000"]}

from openai import OpenAI
import json

client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 定义抽取字段
schema = ['日期', '股票名称', '开盘价', '收盘价', '成交量']

# 示例
example_data = [
    {
        "content": "2023-01-10，股市震荡。股票强大科技A股今日开盘价100人民币，一度飙升至105人民币，随后回落至98人民币，最终以102人民币收盘，成交量达到520000。",
        "answers": {
            "日期": "2023-01-10",
            "股票名称": "强大科技A股",
            "开盘价": "100人民币",
            "收盘价": "102人民币",
            "成交量": "520000"
        }
    },
    {
        "content": "2024-05-16，股市利好。股票英伟达美股今日开盘价105美元，一度飙升至109美元，随后回落至100美元，最终以116美元收盘，成交量达到3560000。",
        "answers": {
            "日期": "2024-05-16",
            "股票名称": "英伟达美股",
            "开盘价": "105美元",
            "收盘价": "116美元",
            "成交量": "3560000"
        }
    }
]

# 待抽取文本
questions = [
    "2025-06-16，股市利好。股票传智教育A股今日开盘价66人民币，一度飙升至70人民币，随后回落至65人民币，最终以68人民币收盘，成交量达到123000。",
    "2025-06-06，股市利好。股票黑马程序员A股今日开盘价200人民币，一度飙升至211人民币，随后回落至201人民币，最终以206人民币收盘。"
]

# 构造messages
messages = [
    {"role": "system", "content": f"你帮我完成信息抽取，我给你句子，你抽取{schema}信息，按JSON字符串输出，如果某些信息不存在，用'原文未提及'表示，请参考如下示例："}
]
for example in example_data:
    messages.append({"role": "user", "content": example["content"]})
    messages.append({"role": "assistant", "content": json.dumps(example["answers"], ensure_ascii=False)})

for q in questions:
    response = client.chat.completions.create(
        model="qwen3-max",
        messages=messages + [{"role": "user", "content": f"按照示例，抽取这段文字的信息：{q}"}]
    )
    print(response.choices[0].message.content)
5.文本匹配prompt设计

在文本分类的prompt设计中，主要考虑两点，一，需要向模型解释什么叫作「文本匹配任务」；二， 让模型按照我们指定的格式输出，即借用 FewShot 的方式，给模型展示一些正确的例子：

User:
句子一：公司ABC发布了季度财报，显示盈利增长。
句子二：财报披露，公司ABC利润上升

Bot:
是

User:
句子一：黄金价格下跌，投资者抛售。
句子二：外汇市场交易额创新高

Bot:
不是

from openai import OpenAI


client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

examples_data = {
    "是": [
        ("公司ABC发布了季度财报，显示盈利增长。", "财报披露，公司ABC利润上升。"),
        ("公司ITCAST发布了年度财报，显示盈利大幅度增长。", "财报披露，公司ITCAST更赚钱了。")
    ],
    "不是": [
        ("黄金价格下跌，投资者抛售。", "外汇市场交易额创下新高。"),
        ("央行降息，刺激经济增长。", "新能源技术的创新。")
    ]
}

questions = [
    ("利率上升，影响房地产市场。", "高利率对房地产有一定的冲击。"),
    ("油价大幅度下跌，能源公司面临挑战。", "未来智能城市的建设趋势越加明显。"),
    ("股票市场今日大涨，投资者乐观。", "持续上涨的市场让投资者感到满意。")
]

messages = [
    {"role": "system", "content": f"你帮我完成文本匹配，我给你2个句子，被[]包围，你判断它们是否匹配，回答是或不是，请参考如下示例："}
]

for key, value in examples_data.items():
    for v in value:
        messages.append({"role": "user", "content": f"句子1：[{v[0]}], 句子2：[{v[1]}]"})
        messages.append({"role": "assistant", "content": key})

for q in questions:
    response = client.chat.completions.create(
        model="qwen3-max",
        messages=messages + [{"role": "user", "content": f"句子1：[{q[0]}], 句子2：[{q[1]}]"}]
    )
    print(response.choices[0].message.content)
五、LangChain框架环境部署

win+R进入终端，输入：

pip install langchain langchain-community langchain-ollama dashscope chromadb -i https://pypi.tuna.tsinghua.edu.cn/simple
六、RAG相关

LLM存在问题：知识非实时，模型训练好后不具备自动更新知识的能力，会导致部分信息滞后；领域知识缺乏，大模型的知识来源于训练数据，这些数据主要来自公开的互联网和开源数据集，无法覆盖特定领域或高度专业化的内部知识；幻觉问题，有时会在回答中生成看似合理但实际上是错误的信息；信息安全等。

RAG（Retrieval-Augmented Generation）即检索增强生成，为大模型提供了从特定数据源检索到的信息，以此来修正和补充生成的答案。

RAG可以总结为一个公式：RAG = 检索技术 + LLM 提示

RAG工作流程：先检索，再生成。不是让大模型只靠自身记忆回答，而是先从外部知识库里找到相关资料，再把这些资料连同用户问题一起交给大模型生成答案。即用户提出问题->系统先做索引（包括文档切割、文本嵌入、建立索引库）->对用户问题做检索（包括把用户问题也也转成向量、在知识库里查找最相关的文本块）->Prompt组合（问题+相关文档快）->LLM 生成答案->把答案返回给用户


RAG工作流程

在线流程与离线流程
RAG标准流程：由索引（Indexing）、检索（Retriever）和生成（Generation）三个核心阶段组成。

索引阶段，通过处理多种来源多种格式的文档提取其中文本，将其切分为标准长度的文本块（chunk），并进行嵌入向量化（embedding），向量存储在向量数据库（vector database）中。

加载文件
内容提取
文本分割，形成 chunk
文本向量化
存入向量数据库
检索阶段，用户输入的查询（query）被转化为向量表示，通过相似度匹配从向量数据库中检索出最相关的文本块。

query 向量化
在文本向量中匹配出与 query 向量相似的 top_k
生成阶段，检索到的相关文档与查询共同构成提示词（Prompt），输入大语言模型（LLM），生成精确且具备上下文关联的回答。

匹配出的文本作为上下文和问题一起添加到 prompt 中
提交给 LLM 生成答案

RAG标准流程
LangChain支持大语言模型、聊天模型、嵌入模型。

LangChain调用大语言模型及其流式输出：

from langchain_community.llms.tongyi import Tongyi

model=Tongyi(model="qwen-max") # 得到模型对象
print(model.invoke(input="你是谁？简洁回答。")) # 调用模型（一次性返回结果）

res=model.stream(input="你是谁？简洁回答。") # 调用模型（流式返回结果）
for chunk in res:
    print(chunk, end="", flush=True)
LangChain调用聊天模型及静态创建消息对象：

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

chat=ChatTongyi(model="qwen-max")

# 静态，直接创建message类对象
messages = [
    SystemMessage(content="你是一名来自边塞的诗人"),
    HumanMessage(content="给我写一首唐诗"),
    AIMessage(content="锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。"),
    HumanMessage(content="基于你上一首的格式，再来一首")
]

res=chat.stream(input=messages)

for chunk in res:
    print(chunk.content, end="", flush=True) # 是chunk.content，不是chunk
消息包含HumanMessage, SystemMessage, AIMessage三种类型

LangChain调用聊天模型及动态创建消息对象：

from langchain_community.chat_models.tongyi import ChatTongyi

chat=ChatTongyi(model="qwen-max")

# 动态，运行时langchain内部机制将其转换成类对象，支持{}值的替换
messages = [
    ("system", "你是一名来自边塞的诗人"),
    ("human", "给我写一首唐诗"),
    ("ai", "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。"),
    ("human", "基于你上一首的格式，再来一首")
]

res=chat.stream(input=messages)

for chunk in res:
    print(chunk.content, end="", flush=True)
可做到消息简写。

LangChain调用文本嵌入模型：

from langchain_community.embeddings import DashScopeEmbeddings

model=DashScopeEmbeddings()

# 调用模型
print(model.embed_query("我喜欢你"))
print(model.embed_documents(["我喜欢你", "我稀罕你", "晚上吃啥"]))
zeroshot通用提示词模板：

PromptTemplate 表示提示词模板，可以构建一个自定义的基础提示词模板，支持变量的注入，最终生成所需的提示词。

from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi

# zero shot
prompt_template=PromptTemplate.from_template("我的邻居姓{lastname}，刚生了{gender}，你帮我起个古风的名字。简单回答。")
prompt_text=prompt_template.format(lastname="章", gender="女儿") # 调用format注入信息

model=Tongyi(model="qwen-max")
res=model.invoke(input=prompt_text)
print(res)
fewshot通用提示词模板：

参数：

examples：示例数据，list，内套字典
example_prompt：示例数据的提示词模板
prefix：组装提示词，示例数据前内容
suffix：组装提示词，示例数据后内容
input_variables：列表，注入的变量列表

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.llms.tongyi import Tongyi

# 示例的模板
example_template=PromptTemplate.from_template("单词：{word}，反义词：{antonym}")

# 示例的动态数据注入，必须是list套字典
example_data=[
    {"word":"大", "antonym":"小"},
    {"word":"上", "antonym":"下"}
]

few_shot_template=FewShotPromptTemplate(
    example_prompt=example_template,                # 示例数据的模板
    examples=example_data,                          # 示例的动态数据注入
    prefix="给出给定词的反义词，有如下示例：",
    suffix="基于示例告诉我：{input_word}的反义词是？",
    input_variables=['input_word']                  # 声明在前缀或后缀中需要注入的变量名
)

prompt_text = few_shot_template.invoke(input={"input_word":"左"}).to_string()

model=Tongyi(model="qwen-max")
print(model.invoke(input=prompt_text))
模板类foramt方法和invoke方法的区别：

在功能方面，format 主要用于简单的字符串替换，通过解析占位符生成提示词；而 invoke 是基于 Runnable 接口的标准方法，不仅可以解析占位符，还能生成结构化的提示结果。

在返回值上，format 返回的是普通字符串；而 invoke 返回的是 PromptValue 类型对象，更适合在 LangChain 等框架中进一步处理。

在传参方式上，format 使用类似 .format(k=v, k=v, ...) 的形式；而 invoke 则使用字典形式 .invoke({"k": v, "k": v, ...})，结构更加统一和灵活。

在解析能力上，format 只支持解析 {} 这种基础占位符；而 invoke 除了支持 {} 占位符外，还支持如 MessagesPlaceholder 这样的结构化占位符，因此在复杂对话场景中更加强大和通用。



一、ChatPromptTemplate

聊天提示词模板支持任意历史会话信息的注入。
方法：在聊天的基础模板中，通过.from_messages方法从列表中获取多轮次会话。而由于历史会话信息是多轮次积攒的，非静态的，故需要MessagesPlaceholder动态注入。

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatTongyi

chat_prompt_template=ChatPromptTemplate.from_messages(      # 接收列表
    [
        ("system", "你是一个边塞诗人"),
        MessagesPlaceholder("history"),
        ("human", "请再来一首五言绝句"),
    ]
)
history_data=[
    ("human", "我要七言绝句")
]
prompt_text=chat_prompt_template.invoke({"history": history_data}).to_string()

model=ChatTongyi(model="qwen3-max")
res=model.invoke(prompt_text)
print(res.content)
MessagesPlaceholder：模板里的占位符，专门放历史消息

chat_prompt_template.invoke({"history": history_data})把 history_data 填进模板里，生成一个ChatPromptValue，再通过to_string()转成字符串，发送给模型进行回答

二、链的基础使用

LangChain链的核心工作原理：将组件串联，上一个组件的输出作为下一个组件的输入。

chain = prompt_template | model
要注意，核心前提是Runnable子类对象才能入链，不是的需要RunnableLambda转换一下。如下图几个类对象都可以入链：


用链把模板和模型串起来，可以通过invoke和stream两种方式调用：

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatTongyi

chat_prompt_template=ChatPromptTemplate.from_messages( # 接收列表
    [
        ("system", "你是一个边塞诗人"),
        MessagesPlaceholder("history"),
        ("human", "请再来一首五言绝句"),
    ]
)
history_data=[
    ("human", "我要七言绝句")
]
model=ChatTongyi(model="qwen3-max")


chain=chat_prompt_template | model

# invoke
res=chain.invoke({"history": history_data})
print(res.content)

# stream
res=chain.stream({"history": history_data})
for chunk in res:
    print(chunk.content, end="", flush=True)
三、StrOutputParser解析器

AIMessage输入，str输出
模板输出结果是PromptValue类型，模型接收PromptValue和字符串等类型输入，输出AIMessage类型结果，若想将此结果再喂给模型进行问答，需类型转换，即将AIMessage转为字符串：

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

parser = StrOutputParser()
model = ChatTongyi(model="qwen3-max")
prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}，刚生了{gender}，请起名，仅告知我名字无需其它内容。"
)

chain = prompt | model | parser | model | parser

res = chain.invoke({"lastname": "裴", "gender": "女儿"})
print(res)
四、JsonOutputParser解析器

AIMessage输入，字典输出
模板接收字典类型，若想将模型输出结果喂给模板，需要把AIMessage类型转成字典，即dict[json]：

例：先让模型给“简姓女孩”起一个名字，并强制返回 JSON；再把 JSON 里的 name 提取出来，填进第二个提示模板，让模型解释这个名字的含义；最后把解释结果用流式方式打印出来

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

json_parser = JsonOutputParser()
str_parser = StrOutputParser()
model = ChatTongyi(model="qwen3-max")
first_prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}，刚生了{gender}，请帮忙起一个独特、风雅的名字，"
    "并封装为JSON格式返回给我。要求：key是name，value就是你起的名字，请严格遵守格式要求。"
)
second_prompt = PromptTemplate.from_template(
    "姓名：{name}，请帮我简洁地解析含义。"
)

chain = first_prompt | model | json_parser | second_prompt | model | str_parser

res = chain.stream({"lastname": "简", "gender": "女儿"})
for chunk in res:
    print(chunk, end="", flush=True)
json_parser把模型输出的 JSON 文本解析成 Python 字典{"name": "简疏桐"}，第二个提示模板接收这个字典，并取name。

五、自定义函数入链

自己编写Lambda匿名函数完成自定义逻辑的数据转换。

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables import RunnableLambda

model = ChatTongyi(model="qwen3-max")
str_parser = StrOutputParser()
first_prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}，刚生了{gender}，请帮忙起名字，仅告知我名字，不要额外信息。"
)

second_prompt = PromptTemplate.from_template(
    "姓名{name}，请帮我解析含义。"
)

# 匿名函数，等价于：
# def my_func(ai_msg):
#     return {"name": ai_msg.content}
my_parser = RunnableLambda(lambda ai_msg:{"name":ai_msg.content})

chain = first_prompt | model | my_parser | second_prompt | model | str_parser

res = chain.stream({"lastname":"简", "gender":"男孩"})
for chunk in res:
    print(chunk,end="", flush=True)
六、会话记忆

1.临时会话记忆

在原有prompt|model|parser的基础上，加了一层 RunnableWithMessageHistory，让链具备两个能力：

每次调用前，自动读取该会话之前的聊天记录
每次调用后，自动把本轮问答写回聊天记录
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi

str_parser = StrOutputParser()
model = ChatTongyi(model="qwen3-max")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "根据会话历史回复用户问题。会话历史："),
        MessagesPlaceholder("chat_history"), # 这个位置将来会自动插入历史对话消息
        ("human", "回答如下问题：{input}")
    ]
)

# 原有链
base_chain = prompt | model | str_parser

# 按会话id取聊天记录；没有就新建，有就直接拿现成的
# {
#     "user1": <这次会话/这位用户的聊天记录对象>,
#     "user2": <另一次会话/另一位用户的聊天记录对象>
# }
# session_id和InMemoryChatMessageHistory类示例一一对应
store={}
def get_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 创建一个新链，增强原有链功能（带记忆的base_chain）：执行时自动读取历史消息，并把本轮新消息写回历史
conversation_chain = RunnableWithMessageHistory(
    base_chain,                                # 原始链
    get_history,                               # 函数，用来根据会话id获取该会话对应的InMemoryChatMessageHistory类对象）
    input_messages_key="input",                # 指“这一轮用户新输入在哪个字段里”
    history_messages_key="chat_history"        # 指“历史消息要塞到prompt的哪个位置里”
)

if __name__ == '__main__':
    # 固定格式，为当前程序配置session_id
    session_config={
        "configurable":{
            "session_id":"user_001"
        }
    }

    res = conversation_chain.invoke({"input": "小明有两只猫"}, session_config)
    print("第一次执行", res)
    res = conversation_chain.invoke({"input": "小红有一只狗"}, session_config)
    print("第二次执行", res)
    res = conversation_chain.invoke({"input": "共有几只宠物？"}, session_config)
    print("第三次执行", res)
2.长期会话记忆

在上一版“内存记忆”的基础上，进一步实现了：把某个session_id的聊天记录持久化保存到本地文件中。

"""把某个session_id对应的聊天记录保存到本地文件里"""

import os, json
from typing import Sequence
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import message_to_dict, messages_from_dict, BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

# 写入文件时：                            读取文件时：
# Sequence[BaseMessage]                 文件中的JSON字符串
# → list[BaseMessage]                   → list[dict]
# → list[dict]                          → list[BaseMessage]
# → JSON字符串                           → 返回给 LangChain
# → 写入文件

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        self.session_id = session_id                                            # 会话id
        self.storage_path = storage_path                                        # 不同会话id存储文件所在的文件夹路径
        self.file_path = os.path.join(self.storage_path, self.session_id)       # 完整文件路径
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)             # 确保文件夹是存在的

    # 追加聊天消息并写入文件
    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        all_message = list(self.messages)                                       # 已有消息列表
        all_message.extend(messages)                                            # 追加新的
        new_messages = [message_to_dict(message) for message in all_message]    # BaseMessage对象不能直接写进JSON文件，要先把BaseMessage消息转成dict字典
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f)                                          # 再写进JSON文件

    # 从文件中读取全部聊天记录
    @property                                                                   # 表示messages是属性而非函数，当LangChain需要历史消息时，它会去访问history.messages
    def messages(self) -> list[BaseMessage]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)
                return messages_from_dict(messages_data)                        # 把JSON的list[dict]字典列表转换回list[BaseMessage]消息对象列表
        except FileNotFoundError:
            return []

    # 清空聊天记录
    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)



str_parser = StrOutputParser()
model = ChatTongyi(model="qwen3-max")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "根据会话历史回复用户问题。会话历史："),
        MessagesPlaceholder("chat_history"),
        ("human", "回答如下问题：{input}")
    ]
)
base_chain = prompt | model | str_parser

def get_history(session_id):
    return FileChatMessageHistory(session_id, "./chat_history")

conversation_chain = RunnableWithMessageHistory(    # RunnableWithMessageHistory自动做两件事：1.调用链前，根据session_id读取历史消息2.调用链后，把这轮用户消息和AI回复追加保存
    base_chain,
    get_history,    # 用于根据session_id获取某个会话的历史消息存储对象
    input_messages_key="input",
    history_messages_key="chat_history"
)

if __name__ == '__main__':

    session_config={    # 配置session_id
        "configurable":{
            "session_id":"user_001"
        }
    }

    res = conversation_chain.invoke({"input": "小明有两只猫"}, session_config)
    print("第一次执行", res)
    res = conversation_chain.invoke({"input": "小红有一只狗"}, session_config)
    print("第二次执行", res)
    res = conversation_chain.invoke({"input": "共有几只宠物？"}, session_config)
    print("第三次执行", res)
自定义了一个 FileChatMessageHistory 类，用来把聊天记录保存到本地文件。

七、基于向量检索构建提示词模板（最小版RAG）

最基础的 RAG 思路：

先从“知识库”里检索相关资料，再把检索结果连同用户问题一起交给大模型回答
核心流程：

建向量库 → 存资料 → 相似度检索 → 把检索结果塞进 prompt → 模型生成答案

from langchain_community.chat_models import ChatTongyi
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatTongyi(model="qwen3-max")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "以我提供的已知参考资料为主，简洁并专业地回答用户问题。参考资料:{context}。"),
        ("user", "用户提问: {input}")
    ]
)

# 创建一个内存向量库
vector_store = InMemoryVectorStore(embedding=DashScopeEmbeddings(model="text-embedding-v4"))
vector_store.add_texts([
    "减肥就是要少吃多练",
    "在减脂期间吃东西很重要，清淡少油控制卡路里摄入并运动起来",
    "跑步是很好的运动"
])  # 相当于给系统准备了一个很小的“知识库”
input_text = "怎么减肥？"
result = vector_store.similarity_search(input_text, k=2)    # 先检索，再找最相关的2条资料

reference_text = "["
for doc in result:
    reference_text += doc.page_content
reference_text += "]"

chain = prompt | model | StrOutputParser()
res = chain.invoke({"input": input_text, "context": reference_text})
print(res)
创建内存向量库时同时创建了嵌入模型，作用是把文本转成向量。

八、RunnablePassThrough的使用

把“检索”这一步也接进了链里，让用户输入（input_text）进入链后，自动分流成两路：一路原样作为问题 input，一路先去检索知识库并格式化成 context，最后再一起送给模型。

from langchain_community.chat_models import ChatTongyi
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatTongyi(model="qwen3-max")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "以我提供的已知参考资料为主，简洁并专业地回答用户问题。参考资料:{context}"),
        ("user", "用户提问: {input}")
    ]
)

vector_store = InMemoryVectorStore(
    embedding=DashScopeEmbeddings(model="text-embedding-v4")
)
vector_store.add_texts([    # 给向量库准备了几条知识
    "减肥就是要少吃多练",
    "在减脂期间吃东西很重要，清淡少油控制卡路里摄入并运动起来",
    "跑步是很好的运动"
])
input_text = "怎么减肥？"
retriever = vector_store.as_retriever(search_kwargs={"k": 2})   # 把向量库转成了一个检索器，输入字符串，输出list[document]

def format_func(docs: list[Document]):  # 把检索出来的结果（Document列表），转成一个字符串，方便塞进prompt
    if not docs:
        return "无相关参考资料"
    formatted_str = "["
    for doc in docs:
        formatted_str += doc.page_content
    formatted_str += "]"
    return formatted_str

chain = (
    {"input": RunnablePassthrough(), "context": retriever | format_func}    # RunnablePassthrough()表示把原始输入直接原样传过去
    | prompt
    | model
    | StrOutputParser()
)
res = chain.invoke(input_text)
print(res)

"""
retriever:
    - 输入：用户的提问                  str
    - 输出：向量库的检索结果             list[Document]
prompt:
    - 输入：用户的提问 + 向量库的检索结果      dict
    - 输出：完整的提示词                    PromptValue
"""
流程：

先准备一个小型知识库
把知识库存进向量库
用户输入问题
自动去向量库检索最相关的资料
把资料整理成 context
再和原始问题一起交给模型回答
无需手动写similarity_search，手动拼context。 


本次智能客服项目以“某东商品衣服”为例，以衣服属性构建本地知识。使用者可以自由更新本地知识，用户问题的答案也是基于本地知识生成的。

技术栈：LangChain + Chroma 向量数据库 + Streamlit

一、整体架构
RAG即检索、增强和生成，其主要分为2条线：

离线处理：向私有知识库（向量存储）源源不断添加私有知识文档。
向知识库添加来自未来的知识文档（基于模型训练完成时间）
向模型添加私有知识文档
给出模型参考资料，规避模型幻觉（一本正经的胡说八道）
在线处理：用户提问会先基于私有知识库做检索，获取参考资料，同步组装新提示词询问大模型获取结果。
故本智能客服系统分成两条主线：

1.知识库导入线

把.txt文件上传进来，切分、向量化，存进 Chroma向量数据库，同时用MD5做去重。

2.问答对话线

用户在聊天页提问，系统先去向量库里检索相关知识，再把“检索结果 + 历史对话 + 当前问题”一起交给大模型生成回答。

是一个“上传知识 → 建知识库 → 检索知识 → 大模型回答” 的闭环

二、程序层次
1）界面层
负责和用户交互的 Streamlit 页面：

app_file_uploader.py：知识库上传页。
app_qa.py：聊天问答页。
2）业务服务层
负责核心逻辑：

knowledge_base.py：负责知识入库。
rag.py：负责问答链路。
vector_stores.py：负责封装向量库检索器。
file_history_store.py：负责对话历史存取。
3）配置层
config_data.py：集中放路径、模型名、切分参数、top_k、session_id 等配置。
4）数据层
md5.text：记录已经入库过的文本哈希，避免重复导入。
chroma_db/：向量数据库实际落盘位置，由配置指定。
chat_history/：本地聊天记录目录，由 file_history_store.py 写入。
尺码推荐.txt、洗涤养护.txt、颜色选择.txt：知识库原始业务资料。
三、具体文件
1. app_file_uploader.py

"""

上传知识 →
建知识库 →
检索知识 →
大模型回答

"""

import time
import streamlit as st
from knowledge_base import KnowledgeBaseService



st.title("知识库更新服务")
uploader_file = st.file_uploader(
    "请上传TXT文件",
    type=['txt'],
    accept_multiple_files=False,
)

if "service" not in st.session_state:                       # session_state是一个字典，用于保存当前用户会话的状态
    st.session_state["service"] = KnowledgeBaseService()

if uploader_file is not None:
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024                   # KB

    st.subheader(f"文件名：{file_name}")
    st.write(f"格式：{file_type} | 大小：{file_size:.2f} KB")
    text = uploader_file.getvalue().decode("utf-8")         # 把上传文件的内容解码成字符串 get_value -> bytes -> decode('utf-8')

    with st.spinner("载入知识库中……"):
        time.sleep(1)
        result = st.session_state["service"].upload_by_str(text, file_name)
        st.write(result)
 
用 st.file_uploader() 让用户上传一个 TXT 文件
首次进入页面时，在 session_state 里创建 KnowledgeBaseService
读取上传文件内容，解码成字符串
调用 upload_by_str(text, file_name) 把内容写入知识库
把结果显示在页面上
2. app_qa.py

""" 聊天问答页 """

from rag import RagService
import streamlit as st
import config_data as config



st.title("智能客服")
st.divider()    # 分隔符

if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好，有什么可以帮助你？"}]
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()



prompt = st.chat_input()    # 在页面最下方提供用户输入栏，用户输入内容并回车后，prompt就会拿到字符串，否则prompt为空
if prompt:
    st.chat_message("user").write(prompt)                                       # 立刻在页面输出用户的提问
    st.session_state["message"].append({"role": "user", "content": prompt})     # 把用户消息存到历史记录中

    ai_res_list = []                                                            # 准备存储AI返回的流式内容
    with st.spinner("AI思考中..."):
        res_stream = st.session_state["rag"].chain.stream({"input": prompt}, config.session_config)     # 调用RAG链进行流式回答
        def capture(generator, cache_list):                                     # 一边显示，一边缓存
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk
        st.chat_message("assistant").write_stream(capture(res_stream, ai_res_list))
        st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)}) 
设置页面标题“智能客服”
在 session_state 里维护聊天消息列表 message
首次进入时创建 RagService
用户输入问题后，把问题显示出来并存到消息历史里
调用 rag.chain.stream(...) 以流式方式拿回答
一边输出流式回答，一边把完整回答保存到 session_state
3. config_data.py

md5_path = "./md5.text"

# Chroma
collection_name = "rag"
persist_directory = "./chroma_db"

# spliter
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
max_split_char_number = 1000        # 文本分割配置阈值


top_k = 1

embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"

session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }
 
md5_path：MD5 记录文件路径
collection_name、persist_directory：Chroma 向量库名字和本地目录
chunk_size、chunk_overlap、separators、max_split_char_number：文本切分参数
top_k：检索返回条数
embedding_model_name、chat_model_name：嵌入模型和聊天模型名
session_config：会话配置，其中 session_id = "user_001"
4. knowledge_base.py

"""

负责知识入库，即把一段文本切分、向量化，存进Chroma向量数据库，同时用MD5避免重复入库

主要包含工具函数、KnowledgeBaseService两部分

"""

import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime


# 工具函数
def get_string_md5(input_str: str, encoding='utf-8'):   # 给整段文本算MD5值
    str_bytes = input_str.encode(encoding=encoding)     # 1.把字符串编码成字节
    md5_obj = hashlib.md5()                             # 2.hashlib创建md5对象
    md5_obj.update(str_bytes)                           # 3.传字节
    md5_hex = md5_obj.hexdigest()                       # 4.转成十六进制
    return md5_hex

def check_md5(md5_str: str):                            # 检查这个MD5是否已经存在
    if not os.path.exists(config.md5_path):             # 1.检查md5文件是否存在，不存在，创建文件，返回F
        open(config.md5_path, 'w', encoding='utf-8').close()
        return False
    else:                                               # 2.md5文件存在，逐行读，去掉空格换行，找到相等字符串返回T，没找到返回F
        for line in open(config.md5_path, 'r', encoding='utf-8').readlines():
            line = line.strip()
            if line == md5_str:
                return True
        return False

def save_md5(md5_str: str):                             # 把新的md5追加写入文件
    with open(config.md5_path, 'a', encoding="utf-8") as f:
        f.write(md5_str + '\n')



class KnowledgeBaseService(object):
    def __init__(self):
        os.makedirs(config.persist_directory, exist_ok=True)    # 如果向量库存储路径不存在则创建，存在则跳过
        self.chroma = Chroma(                                   # 初始化Chroma向量库对象
            collection_name=config.collection_name,             # 数据库表名
            embedding_function=DashScopeEmbeddings(model=config.embedding_model_name),      # 嵌入模型的作用：把文本转成向量
            persist_directory=config.persist_directory,         # 本地存储路径
        )
        self.spliter = RecursiveCharacterTextSplitter(          # 初始化文本切分器对象
            chunk_size=config.chunk_size,                       # 每块最大字符数
            chunk_overlap=config.chunk_overlap,                 # 相邻块之间重叠字符数
            separators=config.separators,                       # 优先按哪些符号切分
            length_function=len,                                # 用Python自带的len函数计算长度
        )

    def upload_by_str(self, data: str, filename):               # 将传入的字符串向量化，存入向量库中
        md5_hex = get_string_md5(data)                          # 1.给字符串算MD5值
        if check_md5(md5_hex):                                  # 2.查重
            return "[跳过]内容已经存在知识库中"
        if len(data) > config.max_split_char_number:            # 3.判断是否需要切分
            knowledge_chunks: list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]
        metadata = {                                            # 4.构造元数据
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "小简",
        }
        self.chroma.add_texts(                                  # 5.写入向量库（DashScopeEmbeddings把每段文本转成向量，再存入Chroma向量库）
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )
        save_md5(md5_hex)                                       # 6.保存md5记录（把当前文本的md5写入md5文件）
        return "[成功]内容已经成功载入向量库"                        # 7.返回成功信息



if __name__ == '__main__':
    service = KnowledgeBaseService()
    r = service.upload_by_str("peek-a-boo", "testfile")
    print(r)
它主要包含两部分：

（1）工具函数
get_string_md5()：给整段文本算 MD5
check_md5()：检查这个 MD5 是否已经存在
save_md5()：把新的 MD5 追加写入文件。
（2）KnowledgeBaseService
初始化时会做两件事：

创建 Chroma 向量库对象
创建文本切分器 RecursiveCharacterTextSplitter。
核心方法 upload_by_str(data, filename) 的流程是：

对整段文本算 MD5
先查重，已存在就跳过
如果文本太长，就按配置切分；否则直接整段作为一个 chunk
给每个 chunk 附加元数据，比如 source、create_time、operator
调用 self.chroma.add_texts(...) 写入向量库
记录 MD5
返回成功信息
5. vector_stores.py

"""

负责封装向量库检索器

即封装一个基于Chroma的向量库服务类，并返回一个retriever

"""

from langchain_chroma import Chroma
import config_data as config


class VectorStoreService(object):
    def __init__(self, embedding):
        self.embedding = embedding              # 嵌入模型
        self.vector_store = Chroma(             # 初始化向量库
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):                    # 获取检索器
        return self.vector_store.as_retriever(search_kwargs={"k": config.top_k})     # 向量库转检索器



if __name__ == '__main__':
    from langchain_community.embeddings import DashScopeEmbeddings
    retriever = VectorStoreService(
        DashScopeEmbeddings(model="text-embedding-v4")
    ).get_retriever()
    res = retriever.invoke("我的体重180斤，尺码推荐")
    print(res)
初始化 Chroma，连接到和知识库写入时相同的 collection_name 与 persist_directory
提供 get_retriever()，把向量库转成 LangChain 的 retriever，并且使用 top_k 控制返回数量
6. file_history_store.py

"""

负责历史对话存取

即把某个session_id对应的聊天记录保存到本地文件里

"""

import json
import os
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict


def get_history(session_id):
    return FileChatMessageHistory(session_id, "./chat_history")

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        self.session_id = session_id
        self.storage_path = storage_path
        self.file_path = os.path.join(self.storage_path, self.session_id)
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        all_messages = list(self.messages)
        all_messages.extend(messages)
        new_messages = [message_to_dict(message) for message in all_messages]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f)

    @property
    def messages(self) -> list[BaseMessage]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)
                return messages_from_dict(messages_data)
        except FileNotFoundError:
            return []

    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)
它定义了：

get_history(session_id)：返回一个 FileChatMessageHistory
FileChatMessageHistory：把某个会话的消息存成本地 JSON 文件。
这个类实现了三个关键能力：

add_messages()：把新消息追加进历史并写文件
messages 属性：从文件读取历史消息
clear()：清空历史
7. rag.py

"""

负责问答链路

即封装一个支持 检索增强生成（RAG）+ 历史对话记忆 的问答服务类

"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi



class RagService(object):                                                       # 四个参数，一个方法
    def __init__(self):
        self.vector_service = VectorStoreService(embedding=DashScopeEmbeddings(model=config.embedding_model_name))
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，简洁、专业地回答用户问题。参考资料:{context}。"),
                ("system", "我提供用户的对话历史记录，如下："),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问：{input}")
            ]
        )
        self.chat_model = ChatTongyi(model=config.chat_model_name)
        self.chain = self.__get_chain()

    def __get_chain(self):                                          # 生成最终的问答链
        retriever = self.vector_service.get_retriever()             # 1.获取检索器
        def format_for_retriever(value: dict) -> str:               # 2.给检索器准备输入
            return value["input"]
        def format_document(docs: list[Document]):                  # 3.格式化文档（拼接成字符串）
            if not docs:
                return "无相关参考资料"
            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"
            return formatted_str
        def format_for_prompt_template(value):                      # 4.组装prompt（把字典整理成prompt_template能识别的格式，填入）
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["history"] = value["input"]["history"]
            new_value["context"] = value["context"]
            return new_value
        base_chain = (                                              # 5.组装主链
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever) | retriever | format_document
            }
            # {
            #     "input": {
            #         "input": "针织毛衣如何保养？",
            #         "history": [...]
            #     },
            #     "context": "检索到的参考资料字符串"
            # }
            | RunnableLambda(format_for_prompt_template) | self.prompt_template
            | self.chat_model
            | StrOutputParser()
        )
        conversation_chain = RunnableWithMessageHistory(            # 6.给主链加历史记忆
            base_chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return conversation_chain                                   # 7.返回对话链



if __name__ == '__main__':

    session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }

    res = RagService().chain.invoke({"input": "针织毛衣如何保养？"}, session_config)
    print(res)
# 原始invoke时其实只传了{"input": "针织毛衣如何保养？"}，但因为外面包了一层RunnableWithMessageHistory，所以真正进入内部base_chain的数据，会被它自动补充"history":[]


# 用户问题
#    ↓
# 读取历史对话（依据：session_id）
#    ↓
# 提取当前问题（"针织毛衣如何保养？"）
#    ↓
# 向量库检索相关文档
#    ↓
# 格式化检索结果
#    ↓
# 组装Prompt（context + history + input）
#    ↓
# 调用大模型
#    ↓
# 返回答案
#    ↓
# 保存本轮消息到本地文件 
RagService 初始化时做了几件关键事：

创建 VectorStoreService
创建 Prompt 模板
创建聊天模型 ChatTongyi
构建最终 chain。
它的 Prompt 很明确，要求模型：

以提供的参考资料为主回答
结合历史对话
回答当前用户问题。
__get_chain() 是整个 RAG 流程的拼装处，逻辑如下：

从 vector_service 拿到 retriever
从输入里提取当前问题
用当前问题去检索向量库
把检索结果格式化成字符串 context
把 input + history + context 整理成 Prompt 所需格式
把 Prompt 送给聊天模型
用 StrOutputParser() 输出纯文本
再用 RunnableWithMessageHistory 包一层，让链路自动读写历史记录
四、文件是怎么相互作用的
1）知识导入链路
运行逻辑是：

app_file_uploader.py
→ 调 KnowledgeBaseService
→ 读取 config_data.py 配置
→ 在 knowledge_base.py 里做 MD5 查重、文本切分、向量化
→ 写入 chroma_db
→ 把 MD5 写入 md5.text

2）问答链路
运行逻辑是：

app_qa.py
→ 调 RagService
→ RagService 通过 VectorStoreService 连接已有的 Chroma 向量库
→ 用用户问题检索相关知识
→ 同时通过 file_history_store.py 读取历史消息
→ 拼成 Prompt
→ 调用大模型生成回答
→ 再把本轮对话写回历史文件

五、两个非常关键的共享点
1）写库和读库共用同一个 Chroma 配置
knowledge_base.py 写入的库，vector_stores.py 读取的库，依赖的是同一组配置：

collection_name = "rag"
persist_directory = "./chroma_db"
2）问答链通过 session_id 绑定历史
app_qa.py 在调用 chain 时传入 config.session_config，其中 session_id 固定为 "user_001"；rag.py 再借助 RunnableWithMessageHistory 调用 get_history(session_id) 实现多轮对话记忆。
