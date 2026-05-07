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

大模型无法感知或改变外部环境，所以我们可以将大模型和一些工具（如文件读写工具、运行终端命令工具）组装在一起，变成一个能感知并改变外部环境的智能程序，这就是Agent。

注意，大模型本身不能调用工具，调用工具的是agent的工具调用组件，大模型只能请求调用工具。

Agent类型多，擅长领域各不相同，如cursor就是一个用于编程的agent。

from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool


@tool(description="查询天气")
def get_weather() -> str:
    return "晴天"


agent = create_agent(
    model=ChatTongyi(model="qwen3-max"),
    tools=[get_weather],
    system_prompt="你是一个聊天助手，可以回答用户问题。",
)

res = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "明天深圳的天气如何？"},
        ]
    }
)

for msg in res["messages"]:
    print(type(msg).__name__, msg.content)
create_agent用于创建一个Agent智能体，有模型、工具、提示词模板三个参数
tool 装饰器用来把普通Python函数注册成Agent可调用的工具
流式输出：

from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool


@tool(description="获取股价，传入股票名称，返回字符串信息")
def get_price(name: str) -> str:
    return f"股票{name}的价格是20元"


@tool(description="获取股票信息，传入股票名称，返回字符串信息")
def get_info(name: str) -> str:
    return f"股票{name}，是一家A股上市公司，专注于IT职业教育。"


agent = create_agent(
    model=ChatTongyi(model="qwen3-max"),
    tools=[get_price, get_info],
    system_prompt="你是一个智能助手，可以回答股票相关问题，记住请告知我思考过程，让我知道你为什么调用某个工具"
)

res = agent.stream(
    {"messages": [{"role": "user", "content": "传智教育股价多少，并介绍一下"}]},
    stream_mode="values"
)

for chunk in res:
    # chunk是agent在某一步执行完之后的当前状态快照（state），是一个字典
    # 每一个chunk都带有历史记录，所以不能直接print(chunk)
    latest_message = chunk['messages'][-1]

    if latest_message.content:
        print(type(latest_message).__name__, latest_message.content)

    try:    # 不是所有消息对象都有tool_calls这个属性
        if latest_message.tool_calls:
            print(f"工具调用： { [tc['name'] for tc in latest_message.tool_calls]  }")
    except AttributeError as e:
        pass
latest_message=chunk['messages'][-1]表示取最新状态快照字典
输出结果：

HumanMessage 传智教育股价多少，并介绍一下
工具调用： ['get_price', 'get_info']
ToolMessage 股票传智教育，是一家A股上市公司，专注于IT职业教育。
AIMessage 传智教育的当前股价是20元。  

此外，传智教育是一家A股上市公司，专注于IT职业教育领域。如果您还有其他问题，欢迎随时提问！

链和智能体的区别
agent运行模式
agent的运行有多种模式，最有名的一种是ReAct模式，即Reasoning and Acting——思考与行动。

ReAct模式

ReAct流程
思考：model决定是否调用工具
行动：agent调用工具
观察：查看工具执行结果
注意，Observation的结果由工具产生，由agent接收并注入上下文，由模型读取后继续思考。

ReAct流程的核心步骤：Thought -> Action -> Observation -> Final Answer

LangChain的Agent对象遵循ReAct框架要求，在执行的过程中会持续的思考、行动、观察。
from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool


@tool(description="获取体重，返回值是整数，单位千克")
def get_weight() -> int:
    return 90


@tool(description="获取身高，返回值是整数，单位厘米")
def get_height() -> int:
    return 172


agent = create_agent(
    model=ChatTongyi(model="qwen3-max"),
    tools=[get_weight, get_height],
    system_prompt="""你是严格遵循ReAct框架的智能体，必须按「思考→行动→观察→再思考」的流程解决问题，
    且**每轮仅能思考并调用1个工具**，禁止单次调用多个工具。
    并告知我你的思考过程，工具的调用原因，按思考、行动、观察三个结构告知我""",
)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "计算我的BMI"}]},
    stream_mode="values"
):
    latest_message = chunk['messages'][-1]

    if latest_message.content:
        print(type(latest_message).__name__, latest_message.content)

    try:
        if latest_message.tool_calls:
            print(f"工具调用： { [tc['name'] for tc in latest_message.tool_calls]  }")
    except AttributeError as e:
        pass


输出结果：

HumanMessage 计算我的BMI
AIMessage **思考**：计算BMI需要体重（千克）和身高（厘米）两个数据。根据公式 BMI = 体重(kg) / (身高(m))²，首先需要获取这两个数值。由于工具调用限制每轮只能使用一个工具，我决定先获取体重数据。

**行动**：调用get_weight工具获取体重。


工具调用： ['get_weight']
ToolMessage 90
AIMessage **观察**：获取到体重为90千克。

**思考**：已经获得体重数据，接下来需要获取身高数据才能计算BMI。根据工具调用规则，现在需要调用get_height工具获取身高。

**行动**：调用get_height工具获取身高。


工具调用： ['get_height']
ToolMessage 172
AIMessage **观察**：获取到身高为172厘米。

**思考**：现在已获得体重（90kg）和身高（172cm）数据，可计算BMI。根据公式需先将身高转换为米（172cm=1.72m），则BMI=90/(1.72²)=90/2.9584≈30.42。

**行动**：无需调用工具，直接计算并返回结果。

您的BMI为30.42，属于肥胖范围（BMI≥30）。建议关注健康状况并咨询专业医生。
代码示例：一个可以自己写贪吃蛇游戏的agent
项目地址：

VideoCode/Agent的概念、原理与构建模式 at main · MarkTechStation/VideoCode
github.com/MarkTechStation/VideoCode/tree/main/Agent%E7%9A%84%E6%A6%82%E5%BF%B5%E3%80%81%E5%8E%9F%E7%90%86%E4%B8%8E%E6%9E%84%E5%BB%BA%E6%A8%A1%E5%BC%8F
运行：


任务手动输入

运行结果


最终输出：


得到final answer

运行游戏

游戏界面
主要程序
agent.py：主程序入口，负责 Agent 运行逻辑、模型调用、工具调用、命令行启动

"""

实现了一个基于 ReAct思想的命令行Agent。

核心逻辑：
1.把用户任务和系统提示词一起发给大模型
2.要求大模型按固定格式输出：
    <thought>...</thought>：思考过程
    <action>...</action>：下一步要调用的工具
    <final_answer>...</final_answer>：最终答案
3.如果模型输出的是action，程序就解析这个动作、调用对应工具，再把工具返回结果作为<observation>发回模型
4.如此循环，直到模型输出final_answer


代码主要分成两部分：
1.ReActAgent类：负责和模型对话、解析模型输出、调用工具、组织循环
2.三个工具函数：
    read_file
    write_to_file
    run_terminal_command
3.main()：程序入口，启动命令行交互

"""

import ast
import inspect
import os
import re
from string import Template
from typing import List, Callable, Tuple

import click
from dotenv import load_dotenv
from openai import OpenAI
import platform

from prompt_template import react_system_prompt_template


class ReActAgent:
    def __init__(self, model: str, tools: List[Callable], project_directory: str):
        self.client = OpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=ReActAgent.get_api_key(),
        )
        self.model = model
        self.tools = {func.__name__: func for func in tools}                # 把工具列表转成字典，方便按名称调用
        # {
        #     "read_file": read_file,
        #     "write_to_file": write_to_file,
        #     "run_terminal_command": run_terminal_command
        # }   这样模型输出 <action>read_file("a.txt")</action> 时，程序就能通过 "read_file" 找到对应函数(?)
        self.project_directory = project_directory                          # 保存项目目录路径，后面渲染系统提示词时会把目录下文件列给模型(？)

    def run(self, user_input: str):
        # 1.构造初始消息
        messages = [
            {"role": "system", "content": self.render_system_prompt(react_system_prompt_template)},   # 渲染后的系统提示词
            {"role": "user", "content": f"<question>{user_input}</question>"}                         # 把用户问题包成<question>...</question>
        ]

        # 2.进入循环
        while True:

            # 3.请求模型
            content = self.call_model(messages)
            # 模型返回如下文本：
            # <thought>我需要先读一下 main.py 文件</thought>
            # <action>read_file("main.py")</action>
            # 或
            # <thought>我已经完成任务</thought>
            # <final_answer>问题已经解决。</final_answer>

            # 4.提取Thought
            # 正则表达式提取模型输出里的<thought>...</thought>
            thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
            if thought_match:
                thought = thought_match.group(1)
                # 打印
                print(f"\n\n💭 Thought: {thought}")

            # 5.检测模型是否输出Final Answer
            final_answer_match = re.search(r"<final_answer>(.*?)</final_answer>", content, re.DOTALL)
            # 是，直接返回
            if final_answer_match:
                return final_answer_match.group(1).strip()

            # 6.否，提取Action并执行
            action_match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
            if not action_match:
                raise RuntimeError("模型未输出 <action>")
            # 解析动作
            action = action_match.group(1)
            tool_name, args = self.parse_action(action)
            # 打印
            print(f"\n\n🔧 Action: {tool_name}({', '.join(args)})")
            # 危险命令（终端命令）需要人工确认
            should_continue = input(f"\n\n是否继续？（Y/N）") if tool_name == "run_terminal_command" else "y"
            if should_continue.lower() != 'y':
                print("\n\n操作已取消。")
                return "操作被用户取消"
            # 执行
            try:
                observation = self.tools[tool_name](*args)
                # 工具根据工具名从self.tools中找到对应函数，并传入参数执行
                # 然后返回一个observation，如：
                # <observation>README.md 的内容是：...</observation>
            except Exception as e:
                observation = f"工具执行错误：{str(e)}"

            # 7.把observation返回给模型
            # 打印
            print(f"\n\n🔍 Observation：{observation}")
            # 把observation追加进对话历史，继续发给模型
            obs_msg = f"<observation>{observation}</observation>"
            messages.append({"role": "user", "content": obs_msg})           # 注意：工具返回结果被当作“用户消息”喂给模型，意思是：把外部环境反馈作为新输入提供给模型

    def get_tool_list(self) -> str:
        """生成工具列表字符串，包含函数签名和简要说明，供系统提示词使用"""
        tool_descriptions = []
        for func in self.tools.values():
            name = func.__name__
            signature = str(inspect.signature(func))        # 函数签名
            doc = inspect.getdoc(func)                      # 简要说明
            tool_descriptions.append(f"- {name}{signature}: {doc}")
        return "\n".join(tool_descriptions)
    # 最后返回：
    # - add(a: int, b: int): 计算两个数之和
    # - search(query: str): 根据关键词搜索内容
    # - save(data: dict, path: str): 保存数据到文件

    def render_system_prompt(self, system_prompt_template: str) -> str:
        """把系统提示模板中的变量替换成真实内容"""
        tool_list = self.get_tool_list()
        file_list = ", ".join(                                              # 对目录中的每个文件名f，都生成它的绝对路径，并用逗号和空格拼接成一个字符串
            os.path.abspath(os.path.join(self.project_directory, f))
            for f in os.listdir(self.project_directory)
        )
        return Template(system_prompt_template).substitute(                 # 把模板中的变量替换成真实值
            operating_system=self.get_operating_system_name(),
            tool_list=tool_list,
            file_list=file_list
        )

    @staticmethod
    def get_api_key() -> str:
        """从.env文件或环境变量中读取API Key"""
        load_dotenv()                                       # 加载.env文件
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("未找到 DASHSCOPE_API_KEY 环境变量，请在 .env 文件中设置。")
        return api_key

    def call_model(self, messages):
        """向模型发送一次对话请求，拿到模型回复，把回复追加到messages里，并返回回复内容"""
        print("\n\n正在请求模型，请稍等...")
        response = self.client.chat.completions.create(                     # 向模型发送请求
            model=self.model,
            messages=messages,
        )
        content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": content})
        return content

    def parse_action(self, code_str: str) -> Tuple[str, List[str]]:
        match = re.match(r'(\w+)\((.*)\)', code_str, re.DOTALL)
        if not match:
            raise ValueError("Invalid function call syntax")

        func_name = match.group(1)
        args_str = match.group(2).strip()

        # 手动解析参数，特别处理包含多行内容的字符串
        args = []
        current_arg = ""
        in_string = False
        string_char = None
        i = 0
        paren_depth = 0
        
        while i < len(args_str):
            char = args_str[i]
            
            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    string_char = char
                    current_arg += char
                elif char == '(':
                    paren_depth += 1
                    current_arg += char
                elif char == ')':
                    paren_depth -= 1
                    current_arg += char
                elif char == ',' and paren_depth == 0:
                    # 遇到顶层逗号，结束当前参数
                    args.append(self._parse_single_arg(current_arg.strip()))
                    current_arg = ""
                else:
                    current_arg += char
            else:
                current_arg += char
                if char == string_char and (i == 0 or args_str[i-1] != '\\'):
                    in_string = False
                    string_char = None
            
            i += 1
        
        # 添加最后一个参数
        if current_arg.strip():
            args.append(self._parse_single_arg(current_arg.strip()))
        
        return func_name, args
    
    def _parse_single_arg(self, arg_str: str):
        """解析单个参数"""
        arg_str = arg_str.strip()
        
        # 如果是字符串字面量
        if (arg_str.startswith('"') and arg_str.endswith('"')) or \
           (arg_str.startswith("'") and arg_str.endswith("'")):
            # 移除外层引号并处理转义字符
            inner_str = arg_str[1:-1]
            # 处理常见的转义字符
            inner_str = inner_str.replace('\\"', '"').replace("\\'", "'")
            inner_str = inner_str.replace('\\n', '\n').replace('\\t', '\t')
            inner_str = inner_str.replace('\\r', '\r').replace('\\\\', '\\')
            return inner_str
        
        # 尝试使用 ast.literal_eval 解析其他类型
        try:
            return ast.literal_eval(arg_str)
        except (SyntaxError, ValueError):
            # 如果解析失败，返回原始字符串
            return arg_str

    def get_operating_system_name(self):
        os_map = {
            "Darwin": "macOS",
            "Windows": "Windows",
            "Linux": "Linux"
        }
        return os_map.get(platform.system(), "Unknown")


def read_file(file_path):
    """用于读取文件内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def write_to_file(file_path, content):
    """将指定内容写入指定文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content.replace("\\n", "\n"))
    return "写入成功"

def run_terminal_command(command):
    """用于执行终端命令"""
    import subprocess
    run_result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return "执行成功" if run_result.returncode == 0 else run_result.stderr

@click.command()
@click.argument('project_directory',
                type=click.Path(exists=True, file_okay=False, dir_okay=True)
                )
def main(project_directory):
    """
    流程：
        获取项目绝对路径
        注册三个工具
        创建Agent
        用户输入任务
        执行Agent
        输出最终答案
    """
    project_dir = os.path.abspath(project_directory)
    tools = [read_file, write_to_file, run_terminal_command]
    agent = ReActAgent(tools=tools, model="qwen3-max", project_directory=project_dir)

    task = input("请输入任务：")
    final_answer = agent.run(task)
    print(f"\n\n✅ Final Answer：{final_answer}")

if __name__ == "__main__":
    main()
ReActAgent ：整个项目的核心类，负责和模型对话、解析模型输出、调用工具、组织循环
run(self, user_input)：ReActAgent类最关键的方法，负责构造初始消息 -> 进入循环 -> 请求模型 -> 提取Thought -> 检测模型是否输出Final Answer -> 否，提取Action并执行 -> 把observation返回给模型
ReActAgent类还包含get_tool_list，render_system_prompt，get_api_key，call_model，parse_action等函数
三个工具函数：
read_file
write_to_file
run_terminal_command
main() 程序入口：启动命令行交互
prompt_template.py ：给模型看的系统提示词模板，定义模型应该怎么回答、怎么调用工具、输出必须长什么样

react_system_prompt_template = """
你需要解决一个问题。为此，你需要将问题分解为多个步骤。对于每个步骤，首先使用 <thought> 思考要做什么，然后使用可用工具之一决定一个 <action>。接着，你将根据你的行动从环境/工具中收到一个 <observation>。持续这个思考和行动的过程，直到你有足够的信息来提供 <final_answer>。

所有步骤请严格使用以下 XML 标签格式输出：
- <question> 用户问题
- <thought> 思考
- <action> 采取的工具操作
- <observation> 工具或环境返回的结果
- <final_answer> 最终答案

⸻

例子 1:

<question>埃菲尔铁塔有多高？</question>
<thought>我需要找到埃菲尔铁塔的高度。可以使用搜索工具。</thought>
<action>get_height("埃菲尔铁塔")</action>
<observation>埃菲尔铁塔的高度约为330米（包含天线）。</observation>
<thought>搜索结果显示了高度。我已经得到答案了。</thought>
<final_answer>埃菲尔铁塔的高度约为330米。</final_answer>

⸻

例子 2:

<question>帮我找一个简单的番茄炒蛋食谱，并看看家里的冰箱里有没有西红柿。</question>
<thought>这个任务分两步。第一步，找到番茄炒蛋的食谱。第二步，检查冰箱里是否有西红柿。我先用 find_recipe 工具找食谱。</thought>
<action>find_recipe(dish="番茄炒蛋")</action>
<observation>简单的番茄炒蛋食谱：将2个鸡蛋打散，2个番茄切块。热油，先炒鸡蛋，盛出。再热油，炒番茄至软烂，加入鸡蛋，放盐调味即可。</observation>
<thought>好的，我已经有食谱了。食谱需要西红柿。现在我需要用 check_fridge 工具看看冰箱里有没有西红柿。</thought>
<action>check_fridge(item="西红柿")</action>
<observation>冰箱检查结果：有3个西红柿。</observation>
<thought>我找到了食谱，并且确认了冰箱里有西红柿。可以回答问题了。</thought>
<final_answer>简单的番茄炒蛋食谱是：鸡蛋打散，番茄切块。先炒鸡蛋，再炒番茄，混合后加盐调味。冰箱里有3个西红柿。</final_answer>

⸻

请严格遵守：
- 你每次回答都必须包括两个标签，第一个是 <thought>，第二个是 <action> 或 <final_answer>
- 输出 <action> 后立即停止生成，等待真实的 <observation>，擅自生成 <observation> 将导致错误
- 如果 <action> 中的某个工具参数有多行的话，请使用 \n 来表示，如：<action>write_to_file("/tmp/test.txt", "a\nb\nc")</action>
- 工具参数中的文件路径请使用绝对路径，不要只给出一个文件名。比如要写 write_to_file("/tmp/test.txt", "内容")，而不是 write_to_file("test.txt", "内容")

⸻

本次任务可用工具：
${tool_list}

⸻

环境信息：

操作系统：${operating_system}
当前目录下文件列表：${file_list}
"""
模板里规定了模型必须使用这些 XML 标签：

<question>
<thought>
<action>
<observation>
<final_answer>
还给了两个示例，告诉模型应该如何一步步思考、调用工具、等待 observation、最后再总结。

另外，它还特别强调了几条规则：

每次回答必须先有 <thought>
第二个标签必须是 <action> 或 <final_answer>
输出 <action> 后必须停止，等待真实 observation
多行参数要写成 \n
文件路径要用绝对路径
目的：规范输出格式、教模型使用工具、减少模型乱编

小结：

程序组成

程序的ReAct时序图（注意，是模型决定action还是final answer）
middleware中间件
中间件对agent的每一步工作加以拦截，在拦截的过程中，就可以完成我们自定义的逻辑。


LangChain中内置了一些基础的中间件，参见：

https://docs.langchain.com/oss/python/langchain/middleware/built-in
docs.langchain.com/oss/python/langchain/middleware/built-in
LangChain在一些关键执行节点预留了“Hooks”——钩子位置[1]，我们可以在这些位置插入自己的逻辑，实现自定义中间件。

自定义中间件可以简单的使用装饰器来定义。

两类六种装饰器：
节点式钩子：在固定执行节点上插入逻辑

before_agent：在 agent 开始运行前触发
适合做初始化、参数检查、日志记录、上下文注入。
after_agent：在 agent 运行结束后触发
适合做结果记录、后处理、统计耗时、统一输出格式。
before_model：在模型真正调用前触发
适合修改 prompt、补充 system message、鉴权、限流。
after_model：在模型调用完成后触发
适合清洗模型输出、解析结果、记录 token 使用量。
针对工具和模型的包装式钩子：把一次调用整体包起来

wrap_model_call：把每次模型调用包起来（模型执行中）
也就是每调用一次 LLM，它都会先经过你的包装逻辑，可以打印日志、统计耗时、捕获模型报错。
wrap_tool_call：把每次工具调用包起来（工具执行中）
比如 agent 调用搜索工具、数据库工具、计算工具时，你都能统一拦截。
节点式钩子更像“流程中的通知点”，包装式钩子更像“整次调用的控制器”
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_agent, after_agent, before_model, after_model, wrap_model_call, \
    wrap_tool_call
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool
from langgraph.runtime import Runtime


@tool(description="查询天气。传入城市名称字符串，返回字符串天气信息")
def get_weather(city: str) -> str:
    return f"{city}天气：晴天"


@before_agent
def log_before_agent(state: AgentState, runtime: Runtime) -> None:
    """

    state：当前agent的状态数据，通常会包含消息历史、上下文等
    runtime：运行时对象，通常包含执行环境相关信息

    """
    print(f"[before agent]agent启动，并附带{len(state['messages'])}消息")


@after_agent
def log_after_agent(state: AgentState, runtime: Runtime) -> None:
    print(f"[after agent]agent结束，并附带{len(state['messages'])}消息")


@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> None:
    """打印“模型即将调用”，并显示此刻传给模型的消息数"""
    print(f"[before_model]模型即将调用，并附带{len(state['messages'])}消息")


@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> None:
    print(f"[after_model]模型调用结束，并附带{len(state['messages'])}消息")


@wrap_model_call
def model_call_hook(request, handler):
    """

    request：这次模型调用的请求对象，里面一般有模型输入等信息
    handler：真正执行模型调用的函数

    把整次模型调用接过来，自己决定怎么放行。
    也就是它能控制：
    1.调用前做什么
    2.调用时是否继续执行
    3.调用后做什么
    4.出错时怎么处理

    """
    print("模型调用前")
    result = handler(request)
    print("模型调用后")
    return result


@wrap_tool_call
def monitor_tool(request, handler):
    print(f"工具执行：{request.tool_call['name']}")
    print(f"工具执行传入参数：{request.tool_call['args']}")
    return handler(request)


agent = create_agent(
    model=ChatTongyi(model="qwen3-max"),
    tools=[get_weather],
    middleware=[log_before_agent, log_after_agent, log_before_model, log_after_model, model_call_hook, monitor_tool]
)

res = agent.invoke({"messages": [{"role": "user", "content": "深圳今天的天气如何呀，如何穿衣"}]})
print("**********\n", res)


大致流程：
用户提问
agent 启动
模型判断要不要用工具
调用天气工具
工具结果回给模型
模型生成最终回答
agent 结束
全部流程：
用户发问
↓
before_agent
↓
第一次模型调用前：
before_model
wrap_model_call（调用前）
真正调用模型
wrap_model_call（调用后）
after_model
↓
模型判断：需要调用 get_weather
↓
wrap_tool_call（调用前）
↓
执行 get_weather("深圳")
↓
wrap_tool_call（调用后）
↓
把工具结果加入消息上下文
↓
第二次模型调用前：
before_model
wrap_model_call（调用前）
真正调用模型
wrap_model_call（调用后）
after_model
↓
模型基于天气结果生成最终回答
↓
after_agent
↓
返回 res

Plan and Excute模式
一种先规划、再执行的智能体工作方式。


注意：agent套agent



时序图
具体流程：


Plan模型给出第一轮执行计划，执行agent执行，并加入历史执行记录

把用户问题、第一轮执行计划、历史执行记录一起交给Re-Plan模型，给出第二轮执行计划

agent执行第二轮执行计划，并加入历史执行记录

把用户问题、第二轮执行计划、历史执行记录一起交给Re-Plan模型，给出第三轮执行计划

agent执行第三轮执行计划，并加入历史执行记录

把用户问题、第三轮执行计划、历史执行记录一起交给Re-Plan模型，给出最终答案
一个简单例子：

用户目标是帮我策划一次去东京的 5 天旅行，Plan and Execute agent可能这样工作：

Plan：

确认预算和出行日期
查询机票和酒店
设计每日行程
推荐交通方案
汇总成旅行计划
Execute：

查机票
查酒店
查景点开放时间
安排行程顺序
输出最终攻略
经过五轮循环后获得最终答案。

项目以大模型为核心，结合RAG检索增强与Agent任务处理能力，一方面从知识库中检索准确的产品与售后信息，支持功能、价格、对比、操作指导、故障排查和维护建议等智能问答；另一方面面向已购用户，对扫地机器人的使用数据进行分析，如清洁频率、耗材状态和错误日志等，自动生成个性化使用报告与优化建议，帮助用户提升设备使用效率和产品价值。

知识库

项目分层
Streamlit 前端

LangChain Agent

工具层

中间件层

RAG 服务层

向量库层

主方法链
app.py
└── ReactAgent().stream_response(query)
    └── LangChain create_agent(...).stream(...)
        ├── before_model: log_before_model(...)
        ├── dynamic_prompt: report_prompt_switch(...)
        ├── 如果模型决定调用工具:
        │   └── wrap_tool_call: monitor_tool(...)
        │       └── handler(request)
        │           ├── answer(query)
        │           │   └── RagSummarizeService.answer(query)
        │           │       ├── retrieve_documents(query)
        │           │       │   └── retriever.invoke(query)
        │           │       └── chain.invoke({"input", "context"})
        │           ├── get_weather(city)
        │           ├── get_user_location()
        │           ├── get_user_id()
        │           ├── get_current_month()
        │           ├── fetch_external_data(user_id, month)
        │           │   └── load_external_data()
        │           └── fill_context_for_report()
        └── 输出最终消息流
分层详解
前端层
只负责 UI 和会话状态：

页面初始化
保存 agent
保存历史消息
把用户输入交给 ReactAgent
用 write_stream 流式显示结果
本身不处理业务逻辑，只是入口。

Agent层
整个业务的中枢。初始化时，它把：

模型 chat_model
系统提示词 load_system_prompts()
tools
middleware
都装配进 create_agent(...)，而 stream_response() 是真正运行 Agent 的地方。

工具层
给 Agent 提供“能力”的地方。工具分四类：

1. RAG 工具

answer(query) → 直接调用全局对象 rag.answer(query) ，即：

answer()
└── RagSummarizeService.answer()
2. 模拟工具

get_weather、get_user_location、get_user_id、get_current_month 都是 mock/stub，因为天气是固定文本、城市/用户ID/月是随机返回。

3. 外部数据工具

fetch_external_data(user_id, month) 会先调用 load_external_data()，把 CSV 风格数据读进内存，再返回某用户某月的记录，即：

fetch_external_data(user_id, month)
└── load_external_data()
    ├── get_abs_path(agent_config["external_data_path"])
    ├── open(file)
    └── 构造 external_data[user_id][month] = {...}
4.特殊工具

fill_context_for_report()只返回一句话，但它的真实作用不在返回值，而在被 middleware 识别后触发上下文切换，实现提示词切换。

中间件层
负责把“普通工具”调用变成“可观测、可切换 prompt”的工具调用。

1. monitor_tool

是 @wrap_tool_call 中间件。流程是：

记录工具名和参数
调 handler(request) 执行真正工具
若工具名是 fill_context_for_report，就设置 request.runtime.context["report"] = True
返回执行结果
异常处理
即：

monitor_tool(request, handler)
└── result = handler(request)
    └── 真正执行某个 tool
然后:
if tool_name == "fill_context_for_report":
    runtime.context["report"] = True

return result
2. log_before_model

在每次模型调用前打印日志，记录消息条数和最后一条消息内容。

3. report_prompt_switch

是 @dynamic_prompt，会读取 runtime.context["report"]：

False → load_system_prompts()
True → load_report_prompts()
所以“报告模式”的实质调用链是：

Agent 先调用 fill_context_for_report()
└── monitor_tool 发现该工具
    └── runtime.context["report"] = True
下一次模型调用前
└── report_prompt_switch()
    └── 改用 report prompt
RAG 服务层
RagSummarizeService 是一个“单轮检索总结器”，即：

单轮检索 + 拼接上下文 + 一次总结生成。

初始化时：

RagSummarizeService.__init__()
├── self.vector_store = VectorStoreService()
├── self.retriever = self.vector_store.get_retriever()
├── self.prompt_text = load_rag_prompts()
├── self.prompt_template = PromptTemplate.from_template(...)
├── self.model = chat_model
└── self.chain = self._init_chain()
_init_chain() 里真正组装链：

PromptTemplate | chat_model | StrOutputParser
answer(query) 的调用链是：

answer(query)
├── retrieve_documents(query)
│   └── retriever.invoke(query)
├── 遍历 documents 拼 context
└── chain.invoke({"input": query, "context": context})
向量库层
VectorStoreService 做两件事：

1. 在线检索
初始化时创建：

Chroma(...)
RecursiveCharacterTextSplitter(...)
get_retriever() 返回：

self.vector_store.as_retriever(search_kwargs={"k": chroma_config["k"]})
也就是供 RagSummarizeService.retrieve_documents() 调用的 retriever。

2. 离线入库
ingest_documents() 的完整调用关系是：

ingest_documents()
├── check_md5(md5)
├── save_md5(md5)
├── get_file_documents(path)
│   ├── txt_loader(path)
│   └── pdf_loader(path)
├── list_files_by_suffix(data_path, allowed_file_type)
├── for each file:
│   ├── get_file_md5(path)
│   ├── check_md5(md5)
│   ├── get_file_documents(path)
│   ├── self.spliter.split_documents(documents)
│   ├── self.vector_store.add_documents(split_document)
│   └── save_md5(md5)
文件处理能力拆在了 file_utils.py：

get_file_md5
list_files_by_suffix
pdf_loader
txt_loader
配置、路径、提示词、模型的底层支撑
1. 路径统一：path_tool.py
get_project_root() 找项目根目录，get_abs_path(relative_path) 负责把相对路径统一转成绝对路径。整个工程的配置、日志、prompt、数据文件定位都依赖它。

2. 配置加载：config_utils.py
四类配置在模块导入时就被加载成全局变量：

rag_config
chroma_config
prompts_config
agent_config。
所以后续别的模块都是直接 from utils.config_utils import rag_config 这种方式拿配置。

3. 提示词加载：prompt_loader.py
提供三种 prompt：

load_system_prompts()
load_rag_prompts()
load_report_prompts()。
调用位置分别是：

Agent 初始化：load_system_prompts()
RAG 初始化：load_rag_prompts()
middleware 动态切换：load_report_prompts() / load_system_prompts()
4. 模型工厂：factory.py
工厂模式统一创建：

chat_model = ChatTongyi(...)
embedding_model = DashScopeEmbeddings(...)。
其下游调用关系很清晰：

chat_model 被 Agent 和 RAG 复用
embedding_model 被 Chroma 向量库使用
5. 日志：logger_utils.py
get_logger() 会创建统一 logger，并同时挂控制台handler和文件handler，避免重复添加 handler。这个 logger 被工具层、中间件层、文件处理层复用。

三条实际业务流程
普通问答流程
用户输入问题
→ app.py 接收 prompt
→ ReactAgent.stream_response(prompt)
→ Agent 用 system prompt 推理
→ 如需知识检索，则调用 answer(query)
→ RagSummarizeService.answer(query)
→ retriever.invoke(query) 检索向量库
→ 组装 context
→ LLM 生成总结
→ 前端流式展示
报告生成流程
用户提出“生成报告”
→ Agent 决定先调用 fill_context_for_report()
→ monitor_tool 拦截后把 runtime.context["report"] = True
→ 下一次模型调用时，report_prompt_switch() 切到 report prompt
→ Agent 再调用 get_user_id() / get_current_month() / fetch_external_data(...)
→ 收集到外部数据后生成报告文本
→ 前端流式输出
知识入库流程
运行 VectorStoreService().ingest_documents()
→ 列出 data 目录下允许的文件
→ 计算 MD5
→ 查重
→ txt/pdf loader 转成 Document
→ splitter 分块
→ add_documents 写入 Chroma
→ 保存 md5


代码实现
utils/path_tool.py
为整个项目统一提供“项目根目录”和“基于项目根目录的绝对路径”，方便在任意模块中访问配置文件、数据文件、日志目录等资源，而不需要手动写死路径

""" 为整个工程提供统一的绝对路径，让整个工程在任何运行位置下都能稳定找到同一份资源 """

import os


def get_project_root() -> str:
    """ 获取工程所在的根目录 """
    # 当前脚本文件路径，即D:\PythonProject\Agent_Project\utils\path_tool.py
    script_path = os.path.abspath(__file__)                # __file__：utils/path_tool.py
    # 当前脚本所在目录，即D:\PythonProject\Agent_Project\utils
    script_dir = os.path.dirname(script_path)
    # 项目根目录，即D:\PythonProject\Agent_Project
    project_dir = os.path.dirname(script_dir)

    return project_dir


def get_abs_path(relative_path: str) -> str:
    """ 传递相对路径，得到绝对路径 """
    project_dir = get_project_root()                       # D:\PythonProject\Agent_Project
    return os.path.join(project_dir, relative_path)        # D:\PythonProject\Agent_Project\config/config.txt


if __name__ == '__main__':
    print(get_abs_path("config/config.txt"))
utils/logger_handler.py
统一创建 logger，让日志能够同时输出到控制台和日志文件，并且控制两边的日志级别与输出格式，方便调试、排查问题和长期留痕

"""

创建一个logger，让日志同时输出到控制台和日志文件

主要完成了三件事：
1.创建日志目录
2.定义统一的日志格式
3.封装一个通用的get_logger()方法

"""

import logging
from utils.path_tool import get_abs_path
import os
from datetime import datetime



LOG_ROOT = get_abs_path("logs")                                 # 日志根目录
os.makedirs(LOG_ROOT, exist_ok=True)                            # 日志根目录存在，ok；不存在，创建
DEFAULT_LOG_FORMAT = logging.Formatter(                         # 定义日志输出格式
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)



def get_logger(
        name: str = "agent",
        console_level: int = logging.INFO,                      # 控制台只显示INFO及以上的日志，不显示DEBUG
        file_level: int = logging.DEBUG,
        log_path = None,
) -> logging.Logger:
    """

    明确目标：
    返回一个可直接使用的logger
    日志同时输出到控制台和文件
    控制台和文件可以设置不同级别
    日志格式统一
    避免重复添加handler，防止日志重复打印

    """
    # 1.获取一个logger对象
    logger = logging.getLogger(name)                            # 注意：同名logger是同一个对象
    # 2.设置logger的总开关级别
    logger.setLevel(logging.DEBUG)                              # 决定哪些级别的日志有资格继续分发给各个handler
    # 3.判断这个logger有没有handler
    # 多次“logger.info/error...”会多次执行get_logger()，本质上是在给一个名为“agent”的logger反复挂handler
    # 而logger收到一条日志后，会把这条日志依次交给它身上的每一个handler去处理（见最下），故会重复打印
    if logger.handlers:
        return logger
    # 4.给logger挂上“控制台控制器”
    console_handler = logging.StreamHandler()                   # 创建console handler
    console_handler.setLevel(console_level)                     # 设置输出级别
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)            # 设置日志格式
    logger.addHandler(console_handler)                          # 挂到logger上
    # 5.给logger挂上“文件控制器”
    if not log_path:
        log_path = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)
    # 6.返回logger对象
    return logger

# 快捷获取日志器（后面直接import logger这个变量就可以，否则用时还要get_logger()）
logger = get_logger()


if __name__ == '__main__':
    logger.info("信息日志")
    logger.error("错误日志")
    logger.warning("警告日志")
    logger.debug("调试日志")
    # logger.info("信息日志")
    # 1.检查logger是否允许INFO级别通过
    # 2.logger创建一条LogRecord，装着这次日志的各种信息
    # 3.logger把这条日志交给它的handlers，每个handler再各自判断一次能否放行
    # 4.handler用formatter把日志格式化
    # 5.各个handler输出
一个很重要的保护逻辑：防止同一个logger被重复挂载handler
utils/config_utils.py
统一读取项目中的多个 YAML 配置文件，并将其解析为 Python 字典，供系统其他模块直接调用。

"""

读配置文件，并把文件内容（k:v）加载成字典返回。

四个函数（四个配置文件）思想：
先通过get_abs_path()获取配置文件的绝对路径
再用open()打开文件读取
然后用yaml.load()解析文件内容
最后返回解析后的Python对象

"""

import yaml
from utils.path_tool import get_abs_path



def load_rag_config(path: str = get_abs_path("config/rag_config.yml"), encoding: str= "utf-8"):
    with open(path, "r", encoding=encoding) as f:             # 以只读模式打开配置文件，文件对象命名为f，用完自动关闭文件
        return yaml.load(f, Loader=yaml.FullLoader)           # 把YAML文件内容解析成Python字典并返回

def load_chroma_config(path: str = get_abs_path("config/chroma_config.yml"), encoding: str= "utf-8"):
    with open(path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_prompts_config(path: str = get_abs_path("config/prompts_config.yml"), encoding: str= "utf-8"):
    with open(path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_agent_config(path: str = get_abs_path("config/agent_config.yml"), encoding: str= "utf-8"):
    with open(path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

rag_config = load_rag_config()          # 返回的是字典
chroma_config = load_chroma_config()
prompts_config = load_prompts_config()
agent_config = load_agent_config()



if __name__ == '__main__':
    print(rag_config["chat_model_name"])
utils/file_utils.py
一个文件处理工具模块，主要提供三类能力：

计算文件 MD5
筛选目录下指定类型的文件
加载 PDF / TXT 文档，转成 LangChain 的 Document 对象
"""

一个文件处理工具模块，提供三类能力：
1.计算文件的MD5值
2.筛选目录下指定类型的文件
3.加载PDF/TXT文档，转成LangChain的Document对象

"""



import os
import hashlib
from utils.logger_utils import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def get_file_md5(path: str):
    # 1.先判断路径是否存在
    if not os.path.exists(path):
        logger.error(f"[md5计算]文件{path}不存在")
        return
    # 2.再判断路径是否是文件
    if not os.path.isfile(path):
        logger.error(f"[md5计算]路径{path}不是文件")
        return
    # 3.创建md5对象
    md5_obj = hashlib.md5()
    # 4.读取文件
    chunk_size = 4096                                           # 4KB分片，避免文件过大，爆内存
    try:
        with open(path, "rb") as f:                             # 二进制模式打开文件
            while chunk := f.read(chunk_size):                  # 循环分片读取
                md5_obj.update(chunk)                           # md5对象更新（把片传进去）
            """
            chunk = f.read(chunk_size)
            while chunk:
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            """
            # 5.把最终结果转成十六进制字符串并返回
            return md5_obj.hexdigest()
    # 6.异常处理
    except Exception as e:
        logger.error(f"计算文件{path}的md5失败，{str(e)}")
        return None


def list_files_by_suffix(path: str, allowed_types: tuple[str]):
    """ 遍历某个目录，筛选出指定后缀类型的文件，并返回路径列表 """
    # 1.判断是不是文件夹
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return tuple()
    # 2.准备文件路径列表
    files = []
    # 3.遍历目录下的所有文件名
    for f in os.listdir(path):
        # 4.按后缀筛选，拼路径、加入路径列表
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))                 # path是文件夹目录，f是目录下的文件的带后缀文件名
    # 5.把列表转成元组返回
    return tuple(files)


def pdf_loader(path: str, passwd=None) -> list[Document]:
    """ 用LangChain的PyPDFLoader把PDF文件加载成Document列表（列表中只有一个document） """
    return PyPDFLoader(path, passwd).load()


def txt_loader(path: str) -> list[Document]:
    return TextLoader(path, encoding="utf-8").load()
utils/prompt_loader.py
负责加载各提示词，从配置文件中读取提示词文件路径，再把对应的prompt文本内容加载到程序里，供 Agent、RAG和报告生成模块使用。

"""

从prompts.yml配置里取出提示词文件路径，再读取对应的提示词文本内容

三个函数（三种提示词）思想：
1.从prompts_config.yml里拿到某个prompt文件路径，并拼成绝对路径
2.打开文件，读取内容
3.返回提示词文本

两个异常：配置项缺失、读取失败

"""
from utils.config_utils import prompts_config
from utils.path_tool import get_abs_path
from utils.logger_utils import logger


def load_system_prompts():
    try:
        system_prompt_path = get_abs_path(prompts_config["main_prompt_path"])
    except KeyError as e:       # 配置项缺失
        logger.error(f"[load_system_prompts]在yaml配置项中没有main_prompt_path配置项")
        raise e                 # 如果这里只写日志，不raise，外部可能以为函数执行成功了，但实际上并没有拿到prompt内容。

    try:
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:      # 文件读取失败
        logger.error(f"[load_system_prompts]解析系统提示词出错，{str(e)}")
        raise e


def load_rag_prompts():
    try:
        rag_prompt_path = get_abs_path(prompts_config["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_prompts]在yaml配置项中没有rag_summarize_prompt_path配置项")
        raise e

    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompts]解析RAG总结提示词出错，{str(e)}")
        raise e


def load_report_prompts():
    try:
        report_prompt_path = get_abs_path(prompts_config["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompts]在yaml配置项中没有report_prompt_path配置项")
        raise e

    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompts]解析报告生成提示词出错，{str(e)}")
        raise e


# 不添加快捷变量，避免大片“导入即报错”


if __name__ == '__main__':
    print(load_report_prompts())

model/factory.py
用工厂模式统一创建聊天模型和向量模型实例，最后分别生成 chat_model 和 embedding_model 供后续 RAG 流程使用

"""

提供模型，放在模型目录下。

用“工厂模式”统一创建聊天模型和向量模型实例，最后分别生成chat_model和embedding_model供后续RAG流程使用

"""

from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_utils import rag_config



class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:    # 返回类型写父类
        pass

class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(model=rag_config["chat_model_name"])

class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_config["embedding_model_name"])

chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingsFactory().generator()
工厂模式核心思想是：

把“对象怎么创建”单独封装起来，而不是在业务代码里到处直接 new / 实例化对象。
比如不直接写：chat_model=ChatTongyi(model="qwen-turbo")，而是写：chat_model = ChatModelFactory().generator()，这样业务层只拿结果，不关心创建细节。
rag/vector_store.py
本质上是一个RAG知识库加载与检索服务类，把本地数据文件读取出来，切分成文本块，生成向量并存入Chroma向量数据库，然后提供检索器。

"""

本质上是一个向量库服务类+知识库服务类。
负责提供检索器、知识入库，后者就是把本地数据文件读成document形式、切分文本块、转成向量存入Chroma向量数据库，同时用md5进行查重。

完成3件事：
1.初始化向量库和文本切分器
2.提供向量库检索器
3.知识入库

"""
from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_utils import chroma_config
from model.factory import embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_utils import get_file_md5, list_files_by_suffix, pdf_loader, txt_loader
from utils.logger_utils import logger
import os


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_config["collection_name"],
            embedding_function=embedding_model,
            persist_directory=chroma_config["persist_directory"],
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap=chroma_config["chunk_overlap"],
            separators=chroma_config["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_config["k"]})

    def ingest_documents(self):
        """

        知识入库
        即获取本地数据文件列表、去重、把文件读成document、分片、转成向量存入向量库

        """
        # 1.准备工作
        # 三个工具方法：查重md5、保存md5、把文件读成document格式
        def check_md5(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_config["md5_store"])):
                open(get_abs_path(chroma_config["md5_store"]), "w", encoding="utf-8").close()
                return False
            with open(get_abs_path(chroma_config["md5_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True
                return False

        def save_md5(md5_for_check: str):
            with open(get_abs_path(chroma_config["md5_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(path: str):
            if path.endswith("txt"):
                return txt_loader(path)
            if path.endswith("pdf"):
                return pdf_loader(path)
            return []

        # 2.获取待处理文件列表
        files_path: list[str] = list_files_by_suffix(
            get_abs_path(chroma_config["data_path"]),       # data_path: data
            tuple(chroma_config["allowed_file_type"]),
        )
        # 3.遍历文件
        for path in files_path:
            # 4.计算文件MD5并查重
            md5 = get_file_md5(path)
            if check_md5(md5):
                logger.info(f"[知识入库]{path}知识库已加载过该内容，跳过")
                continue    # 继续处理下一个文件
            try:
                # 5.把文件读成document格式
                documents = get_file_documents(path)
                # 如果没有有效内容，跳过
                if not documents:
                    logger.warning(f"[知识入库]{path}内无有效文本内容，跳过")
                    continue
                # 6.切分document
                split_document = self.spliter.split_documents(documents)
                # 如果切分后为空，也跳过
                if not split_document:
                    logger.warning(f"[知识入库]{path}分片后没有有效文本内容，跳过")
                    continue
                # 7.把切分后的document写入向量库
                self.vector_store.add_documents(split_document)
                # 8.保存文件md5值
                save_md5(md5)
                # 9.记录成功日志
                logger.info(f"[知识入库]{path} 内容加载成功")
            # 10.异常处理
            except Exception as e:
                logger.error(f"[知识入库]{path}加载失败：{str(e)}", exc_info=True)   # exc_info为True会会把完整堆栈打印出来
                continue


if __name__ == '__main__':
    vs = VectorStoreService()                   # 创建一个向量库服务对象
    vs.ingest_documents()                       # 知识入库
    retriever = vs.get_retriever()              # 获取检索器

    res = retriever.invoke("迷路")               # 检索相关文本
    for r in res:
        print(r.page_content)
        print("-"*20)
rag/rag_service.py
实现了一个最基础的 RAG 问答 + 总结服务：

用户先提问 → 去向量库检索相关资料 → 把“问题 + 检索到的上下文”一起交给大模型 → 输出最终回答
"""

实现最基础的RAG问答 + 总结服务，检索后按严格规则做总结，即用户提问 → 去向量库检索相关资料 → 把input和context一起交给大模型 → 输出最终总结性回答
其中，检索向量库的方法被封装。


是一个单轮检索总结器，后续作为智能体的一个工具函数被使用。(?)

"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model



class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def retrieve_documents(self, query: str) -> list[Document]:
        """ 输入用户问题，返回检索到的文档列表 """
        return self.retriever.invoke(query)

    def answer(self, query: str) -> str:
        # 1.接收用户问题，去向量库检索
        documents = self.retrieve_documents(query)
        # 2.准备一个空的context
        context = ""
        counter = 0
        # 3.遍历检索到的每个文档
        for doc in documents:
            # 4.给每份文档编号，并拼接文档内容和元数据
            counter += 1
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"
        # 4.把用户问题和资料一起交给链执行
        return self.chain.invoke({"input": query, "context": context})


if __name__ == '__main__':
    rag = RagSummarizeService()
    print(rag.answer("小户型适合哪些扫地机器人"))
agent/tools/tools.py
定义了一组可被Agent调用的工具函数，同时提供了一套外部数据加载与查询机制。

""" 定义了一组可被Agent调用的工具函数，同时提供了一套外部数据加载与查询机制 """

import os
from utils.logger_utils import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
import random
from utils.config_utils import agent_config
from utils.path_tool import get_abs_path



rag = RagSummarizeService()
user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010",]
month_array = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]
external_data = {}



@tool(description="从向量库检索并总结")
def answer(query: str) -> str:
    return rag.answer(query)

@tool(description="返回固定天气信息")
def get_weather(city: str) -> str:
    return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"

@tool(description="随机返回用户城市")
def get_user_location() -> str:
    return random.choice(["深圳", "合肥", "杭州"])

@tool(description="随机返回用户ID")
def get_user_id() -> str:
    return random.choice(user_ids)

@tool(description="随机返回月份")
def get_current_month() -> str:
    return random.choice(month_array)

def load_external_data():
    """
    把外部CSV风格文件的数据，一次性读入内存，并整理成便于按“用户 + 月份”查询的嵌套字典
    目标结构：
    {
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        ...
    }
    """
    # 1.懒加载（只有没数据的时候才去读文件，有数据就不加载了）
    if not external_data:
        # 2.获取external_data的路径
        path = get_abs_path(agent_config["external_data_path"])
        # 3.检查文件是否存在
        if not os.path.exists(path):
            raise FileNotFoundError(f"外部数据文件{path}不存在")
        # 4.跳过表头逐行读取文件
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                # 5.按逗号拆分每行，提取字段
                arr: list[str] = line.strip().split(",")

                user_id: str = arr[0].replace('"', "")
                feature: str = arr[1].replace('"', "")
                efficiency: str = arr[2].replace('"', "")
                consumables: str = arr[3].replace('"', "")
                comparison: str = arr[4].replace('"', "")
                month: str = arr[5].replace('"', "")
                # 6.初始化用户字典
                if user_id not in external_data:
                    external_data[user_id] = {}
                # 7.把记录字典存入嵌套用户字典
                external_data[user_id][month] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }

@tool(description="从external records中查询某用户某月数据，如未检索到，返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    load_external_data()
    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""

@tool(description="无参数，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"
agent/tools/middleware.py
中间件，主要做三件事：

监控工具调用
在模型调用前打日志
根据上下文动态切换 system prompt
"""

监控工具
在模型调用前打印日志
根据上下文动态切换prompt

"""



from typing import Callable
from utils.prompt_loader import load_system_prompts, load_report_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_utils import logger



@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,                                       # 本次工具调用请求的数据包，至少封装了工具名、工具参数、运行时上下文
        handler: Callable[[ToolCallRequest], ToolMessage | Command],    # 真正执行工具的函数
) -> ToolMessage | Command:         # ToolMessage：表示普通工具执行结果（大部分handler(request) 返回的都是）
                                    # Command：表示不只是回消息，还要改状态/控流程
    """ 监控工具调用 """
    # 记录这次调用的工具名字和参数
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")
    try:# ***把request传给handler，真正把请求交给底层工具执行***
        result = handler(request)
        # 记录成功日志
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")
        # 如果本次调用的是fill_context_for_report，就在运行时上下文里打一个标记“report = True”
        # report不是框架自带的固定字段，而是这段代码运行过程中自己往上下文里加的业务标记
        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True
        # ***把真正工具的执行结果原样返回***
        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e

@before_model
def log_before_model(
        state: AgentState,          # 整个Agent当前状态
        runtime: Runtime,           # 记录整个运行过程中的上下文和环境信息
):
    """ 在模型调用前打印日志 """
    # 打印消息条数
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息")
    # 打印最后一条消息的类型和内容
    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")
    return None

@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    """ 模型调用前，调用此函数来动态决定这次使用的提示词 """
    is_report = request.runtime.context.get("report", False)
    if is_report:
        return load_report_prompts()
    return load_system_prompts()
agent/react_agent.py
封装一个基于LangChain Agent的智能体类 ReactAgent，接收用户问题后，以流式方式返回模型生成的回答

"""

封装一个基于LangChain Agent的智能体类ReactAgent，接收用户问题后，以流式方式返回模型生成的回答

"""

from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.tools import (answer, get_weather, get_user_location, get_user_id,
                               get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            system_prompt=load_system_prompts(),
            model=chat_model,
            tools=[answer, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def stream_response(self, query: str):
        """ 接收用户问题，把它交给Agent运行，并把Agent产生的内容流式输出 """
        # 把用户问题封装成Agent需要的输入格式：字典
        input_dict = {"messages": [{"role": "user", "content": query}]}
        # 调用Agent的流式执行接口
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()
    for chunk in agent.stream_response("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
