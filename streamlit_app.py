from __future__ import annotations
import re
from typing import Optional, Tuple, List, Union, Literal
import base64
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
import os
#import openai
import graphviz
from dataclasses import dataclass, asdict, is_dataclass
from textwrap import dedent
from streamlit_agraph import agraph, Node, Edge, Config
# import glm
from zhipuai import ZhipuAI
import  time

st.set_page_config(page_title="AI智能思维导图", layout="wide")

COLOR = "cyan"
FOCUS_COLOR = "red"

# openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["ZHIPUAI_API_KEY"] = "220c41bbc0f4d0298741eea2872223b4.ySxGwf8wBtRB4Xbx"
client = ZhipuAI()

@dataclass
class Message:
    """A class that represents a message in a ChatGPT conversation.
    """
    content: str
    role: Literal["user", "system", "assistant"]

    # is a built-in method for dataclasses
    # called after the __init__ method
    def __post_init__(self):
        self.content = dedent(self.content).strip()
# ###
# START_CONVERSATION = [
#     Message("""You are a useful mind map/undirected graph-generating AI that can generate mind maps based on any input or instructions.""", role="system"),
#     Message("""
#         You have the ability to perform the following actions given a request to construct or modify a mind map/graph:
#         1. add(node1, node2) - add an edge between node1 and node2
#         2. delete(node1, node2) - delete the edge between node1 and node2
#         3. delete(node1) - deletes every edge connected to node1

#         Note that the graph is undirected and thus the order of the nodes does not matter
#         and duplicates will be ignored. Another important note: the graph should be sparse,
#         with many nodes and few edges from each node. Too many edges will make it difficult 
#         to understand and hard to read. The answer should only include the actions to perform, 
#         nothing else. If the instructions are vague or even if only a single word is provided, 
#         still generate a graph of multiple nodes and edges that that could makes sense in the 
#         situation. Remember to think step by step and debate pros and cons before settling on 
#         an answer to accomplish the request as well as possible.

#         Here is my first request: Add a mind map about 机器学习 (Machine Learning).
#     """, role="user"),
#     Message("""
#         add("Machine learning","AI")
#         add("Machine learning", "Reinforcement learning")
#         add("Machine learning", "Supervised learning")
#         add("Machine learning", "Unsupervised learning")
#         add("Supervised learning", "Regression")
#         add("Supervised learning", "Classification")
#         add("Unsupervised learning", "Clustering")
#         add("Unsupervised learning", "Anomaly Detection")
#         add("Unsupervised learning", "Dimensionality Reduction")
#         add("Unsupervised learning", "Association Rule Learning")
#         add("Clustering", "K-means")
#         add("Classification", "Logistic Regression")
#         add("Reinforcement learning", "Proximal Policy Optimization")
#         add("Reinforcement learning", "Q-learning")
#     """, role="assistant"),
#     Message("""
#         Remove the parts about reinforcement learning and K-means.
#     """, role="user"),
#     Message("""
#         delete("Reinforcement learning")
#         delete("Clustering", "K-means")
#     """, role="assistant")
# ]
# ########
# Socket Server和client之间出现断连
START_CONVERSATION = [
    Message("""You are a useful mind map/undirected graph-generating AI that can generate mind maps based on any input or instructions.""", role="system"),
    Message("""
        You have the ability to perform the following actions given a request to construct or modify a mind map/graph:
        1. add(node1, node2) - add an edge between node1 and node2
        2. delete(node1, node2) - delete the edge between node1 and node2
        3. delete(node1) - deletes every edge connected to node1

        Note that the graph is undirected and thus the order of the nodes does not matter
        and duplicates will be ignored. Another important note: the graph should be sparse,
        with many nodes and few edges from each node. Too many edges will make it difficult 
        to understand and hard to read. The answer should only include the actions to perform, 
        nothing else. If the instructions are vague or even if only a single word is provided, 
        still generate a graph of multiple nodes and edges that that could makes sense in the 
        situation. Remember to think step by step and debate pros and cons before settling on 
        an answer to accomplish the request as well as possible.请不要在回答的字符串间使用半角的括号,这样我无法解析字符串。

        Here is my first request: Add a mind map about 机器学习.
            
    """, role="user"),
    Message("""
        add("机器学习","人工智能 (AI) 算不算机器学习的一种形式？")
        add("机器学习","机器学习的主要分支有哪些？")
        add("机器学习", "强化学习")
        add("机器学习", "监督学习")
        add("机器学习", "无监督学习")
        add("监督学习", "监督学习常用于解决什么类型的问题？")
        add("监督学习", "回归和分类有什么区别？")
        add("无监督学习", "我们如何通过非监督学习发现数据中的模式？")
        add("我们如何通过非监督学习发现数据中的模式？", "聚类分析能帮助我们做什么？")
        add("我们如何通过非监督学习发现数据中的模式？", "K 均值算法是聚类分析中常用的方法之一，它有什么优缺点？")
        add("我们如何通过非监督学习发现数据中的模式？", "异常检测在哪些方面有用？")
        add("我们如何通过非监督学习发现数据中的模式？", "降维有助于解决什么问题？")
        add("我们如何通过非监督学习发现数据中的模式？", "关联规则学习可以用于哪些场景？")
        add("强化学习", "强化学习如何通过试错来让模型学习？")
        add("强化学习", "Q学习的工作原理是什么？")
        add("聚类分析能帮助我们做什么？", "K 均值算法是聚类分析中常用的方法之一，它有什么优缺点？")
        add("强化学习如何通过试错来让模型学习？", "我们可以使用哪些算法来实现强化学习？")
        add("强化学习如何通过试错来让模型学习？", "近端策略优化是一种怎样的算法？")
    """, role="assistant"),
    Message("""
        Remove the parts about reinforcement learning and K-means.
    """, role="user"),
    Message("""
        delete("强化学习 (Reinforcement Learning)")
        delete("聚类分析 (Clustering) 能帮助我们做什么？", "K-means")
    """, role="assistant")
]

def conversation_to_message_list(conversation: str) -> List[Message]:
    # 解析字符串为Message对象的列表
    messages = []
    for line in conversation.split("\n"):
        if not line:
            continue  # 跳过空行
        content, role = line.split(" - ")
        messages.append(Message(content, role))
    return messages

def print_with_typewriter_effect(text, delay=0.05):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)

def ask_chatgpt(conversation: List[Message]) -> Tuple[str, List[Message]]:
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     # asdict comes from `from dataclasses import asdict`
    #     messages=[asdict(c) for c in conversation]
    # )

    # print("111:", [asdict(c) for c in conversation], "\n")
    # messages = [] 
    # for c in conversation:
    #         messages.append({"content": c.content, "role": c.role})
    response = client.chat.completions.create(
        model="glm-4",
        messages=[asdict(c) for c in conversation],
        top_p=0.7,
        temperature=0.9,
        stream=False,
        max_tokens=5000,
    )
    # 使用生成的文本
    generated_text = response.choices[0].message.content
    # print("222:", response.choices[0].message.content, "\n")
    print(generated_text)
    print("Token Usage: ", response.usage.total_tokens)

    # turn into a Message object
    # msg = Message(**response["choices"][0]["message"])

    # if response:
    #     for chunk in response:
    #         content1 = chunk.choices[0].delta.content
            # for char in content1:
            #     # my_list = [content1] 
            #     # for _ in range(10): my_list.append(content1)
            #     print(char, end='', flush=True)
            # print_with_typewriter_effect(content1)

    # return the text output and the new conversation
    # return msg.content, conversation + [msg]
    return generated_text, conversation + [generated_text]

def ask_chatgpt2(conversation: List[Message]) -> Tuple[str, List[Message]]:
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     # asdict comes from `from dataclasses import asdict`
    #     messages=[asdict(c) for c in conversation]
    # )

    # print("111:", [asdict(c) for c in conversation], "\n")
    # messages = [] 
    # for c in conversation:
    #         messages.append({"content": c.content, "role": c.role})
    # messages = []
    # for c in conversation[:-1]:  # Exclude the last element (user instruction)
    #     messages.append({"content": c.content, "role": c.role})
    print("ask_chatgpt2:\n", conversation)
    # 检查 conversation 是否为字符串类型
    # if isinstance(conversation, str):
    #     print("Str: change to mssage")
    #     conversation = conversation_to_message_list(conversation)
    # else:
    #     print("Not Str: Do nothing")


    messages = []
    for c in conversation[:-2]:  # Exclude the last two elements (user instruction and previous message)
        messages.append({"content": c.content, "role": c.role})

    response = client.chat.completions.create(
        model="glm-4",
        messages=messages,#[asdict(c) for c in conversation], #
        top_p=0.7,
        temperature=0.9,
        stream=False,
        max_tokens=5000,
    )
    # 使用生成的文本
    generated_text = response.choices[0].message.content
    # print("222:", response.choices[0].message.content, "\n")
    print(generated_text)
    print("Token Usage: ", response.usage.total_tokens)

    # turn into a Message object
    # msg = Message(**response["choices"][0]["message"])

    # if response:
    #     for chunk in response:
    #         content1 = chunk.choices[0].delta.content
            # for char in content1:
            #     # my_list = [content1] 
            #     # for _ in range(10): my_list.append(content1)
            #     print(char, end='', flush=True)
            # print_with_typewriter_effect(content1)

    # return the text output and the new conversation
    # return msg.content, conversation + [msg]
    return generated_text, conversation + [generated_text]

class MindMap:
    """A class that represents a mind map as a graph.
    """
    
    def __init__(self, edges: Optional[List[Tuple[str, str]]]=None, nodes: Optional[List[str]]=None) -> None:
        self.edges = [] if edges is None else edges
        self.nodes = [] if nodes is None else nodes
        self.save()

    @classmethod
    def load(cls) -> MindMap:
        """Load mindmap from session state if it exists
        
        Returns: Mindmap
        """
        if "mindmap" in st.session_state:
            return st.session_state["mindmap"]
        return cls()

    def save(self) -> None:
        # save to session state
        st.session_state["mindmap"] = self

    def is_empty(self) -> bool:
        return len(self.edges) == 0
    
    def ask_for_initial_graph(self, query: str) -> None:
        """Ask GLM to construct a graph from scrach.
        Args:
            query (str): The query to ask GLM about.
        Returns:
            str: The output from GLM.
        """

        conversation = START_CONVERSATION + [
            Message(f"""
                Great, now ignore all previous nodes and restart from scratch. I now want you do the following:    

                {query}
            """, role="user")
        ]
        print(conversation)

        # output = """
        #             add("机器学习","数据预处理 (Data Preprocessing)")
        #             add("机器学习","模型选择 (Model Selection)")
        #             add("机器学习","训练 (Training)")
        #             add("机器学习","评估 (Evaluation)")
        #             add("机器学习","部署 (Deployment)")
        #             add("数据预处理 (Data Preprocessing)","数据清洗 (Data Cleaning)")
        #             add("数据预处理 (Data Preprocessing)","特征工程 (Feature Engineering)")
        #             add("模型选择 (Model Selection)","算法选择 (Algorithm Selection)")
        #             add("模型选择 (Model Selection)","超参数调优 (Hyperparameter Tuning)")
        #             add("训练 (Training)","监督学习 (Supervised Learning)")
        #             add("训练 (Training)","无监督学习 (Unsupervised Learning)")
        #             add("评估 (Evaluation)","交叉验证 (Cross-Validation)")
        #             add("评估 (Evaluation)","性能指标 (Performance Metrics)")
        #             add("部署 (Deployment)","模型上线 (Model Deployment)")
        #             add("部署 (Deployment)","监控 (Monitoring)")
        #             add("部署 (Deployment)","维护 (Maintenance)")
        #             """
    #     output = """
    #     add("Machine learning","AI")
    #     add("Machine learning", "Reinforcement learning")
    #     add("Machine learning", "Supervised learning")
    #     add("Machine learning", "Unsupervised learning")
    #     add("Supervised learning", "Regression")
    #     add("Supervised learning", "Classification")
    #     add("Unsupervised learning", "Clustering")
    #     add("Unsupervised learning", "Anomaly Detection")
    #     add("Unsupervised learning", "Dimensionality Reduction")
    #     add("Unsupervised learning", "Association Rule Learning")
    #     add("Clustering", "K-means")
    #     add("Classification", "Logistic Regression")
    #     add("Reinforcement learning", "Proximal Policy Optimization")
    #     add("Reinforcement learning", "Q-learning")
    # """
        
    #     output = """
    #     add("机器学习（222222）","数据预处理")
    #     add("机器学习","模型选择")
    #     add("机器学习","训练")
    #     add("机器学习","评估")
    #     add("机器学习","部署")
    #     add("数据预处理","数据清洗")
    #     add("数据预处理","特征工程")
    #     add("模型选择","算法选择")
    #     add("模型选择","超参数调优")
    #     add("训练","监督学习")
    #     add("训练","无监督学习")
    #     add("评估","交叉验证")
    #     add("评估","性能指标")
    #     add("部署","模型上线")
    #     add("部署","监控")
    #     add("部署","维护")
    # """
    #     output = """
    #     add("机器学习（Machine Learning）","人工智能 （AI） 算不算机器学习的一种形式？")
    #     add("机器学习（Machine Learning）","机器学习的主要分支有哪些？")
    #     add("机器学习（Machine Learning）", "强化学习（Reinforcement Learning）")
    #     add("机器学习（Machine Learning）", "监督学习（Supervised Learning）")
    #     add("机器学习（Machine Learning）", "无监督学习（Unsupervised learning）")
    #     add("监督学习（Supervised Learning）", "监督学习常用于解决什么类型的问题？")
    #     add("监督学习（Supervised Learning）", "回归（Regression） 和 分类（Classification）有什么区别？")
    #     add("无监督学习（Unsupervised learning）", "我们如何通过非监督学习发现数据中的模式？")
    #     add("我们如何通过非监督学习发现数据中的模式？", "聚类分析（Clustering） 能帮助我们做什么？")
    #     add("我们如何通过非监督学习发现数据中的模式？", "K 均值算法（K-means） 是聚类分析中常用的方法之一，它有什么优缺点？")
    #     add("我们如何通过非监督学习发现数据中的模式？", "异常检测（Anomaly Detection）在哪些方面有用？")
    #     add("我们如何通过非监督学习发现数据中的模式？", "降维（Dimensionality Reduction）有助于解决什么问题？")
    #     add("我们如何通过非监督学习发现数据中的模式？", "关联规则学习（Association Rule Learning）可以用于哪些场景？")
    #     add("强化学习（Reinforcement Learning）", "强化学习如何通过试错来让模型学习？")
    #     add("强化学习（Reinforcement Learning）", "Q 学习（Q-learning）的工作原理是什么？")
    #     add("聚类分析（Clustering）能帮助我们做什么？", "K 均值算法（K-means）是聚类分析中常用的方法之一，它有什么优缺点？")
    #     add("强化学习如何通过试错来让模型学习？", "我们可以使用哪些算法来实现强化学习？")
    #     add("强化学习如何通过试错来让模型学习？", "近端策略优化（Proximal Policy Optimization）是一种怎样的算法？")
    #     add("强化学习如何通过试错来让模型学习？", "Q 学习（Q-learning）的工作原理是什么？")
    # """
        output, self.conversation = ask_chatgpt(conversation)
        # feedback_indicator = "请在详细的介绍一下。"
        # conversation.append(Message(feedback_indicator, role="user"))
        # print(conversation)
        output, self.conversation = ask_chatgpt(conversation)
        # replace=True to restart
        self.parse_and_include_edges(output, replace=False)

    def ask_for_extended_graph(self, selected_node: Optional[str]=None, text: Optional[str]=None) -> None:
        """Cached helper function to ask GLM to extend the graph.

        Args:
            query (str): query to ask GLM about
            edges_as_text (str): edges formatted as text

        Returns:
            str: GLM output
        """
        print("------ask_for_extended_graph", text)
        # do nothing
        if (selected_node is None and text is None):
            print("do nothing")
            return

        # change description depending on if a node
        # was selected or a text description was given
        #
        # note that the conversation is copied (shallowly) instead
        # of modified in place. The reason for this is that if
        # the chatgpt call fails self.conversation will not
        # be updated
        conversation = self.conversation.copy()
        print("self.conversation.copy():\n", conversation, "\n")
        if selected_node is not None:
            # prepend a description that this node
            # should be extended
            conversation = self.conversation + [
                Message(f"""
                    add new edges to new nodes, starting from the node "{selected_node}"
                """, role="user")
            ]
            st.session_state.last_expanded = selected_node
            print("prepend a description that this node should be extended:\n", conversation, "\n")
        else:
            # just provide the description
            conversation = self.conversation + [Message(text, role="user")]
            print("just provide the description:\n", conversation, "\n")

        # now self.conversation is updated
        output, self.conversation = ask_chatgpt2(conversation)
    #     output = """
    #     add("机器学习（222222）","数据预处理1111111111111111111111111111111111111111111")
    #     add("机器学习","模型选择")
    #     add("机器学习","训练")
    #     add("机器学习","评估")
    #     add("机器学习","部署")
    #     add("数据预处理","数据清洗")
    #     add("数据预处理","特征工程")
    #     add("模型选择","算法选择")
    #     add("模型选择","超参数调优")
    #     add("训练","监督学习")
    #     add("训练","无监督学习")
    #     add("评估","交叉验证")
    #     add("评估","性能指标")
    #     add("部署","模型上线")
    #     add("部署","监控")
    #     add("部署","维护")
    # """

    #     output = """
    #     add("机器学习（Machine Learning）","人工智能 （AI） 算不算机器学习的一种形式？")
    #     add("机器学习（Machine Learning）","机器学习的主要分支有哪些？")
    #     add("机器学习（Machine Learning）", "强化学习（Reinforcement Learning）")
    #     add("机器学习（Machine Learning）", "监督学习（Supervised Learning）")
    #     add("机器学习（Machine Learning）", "无监督学习（Unsupervised learning）")
    #     add("监督学习（Supervised Learning）", "监督学习常用于解决什么类型的问题？")
    #     add("监督学习（Supervised Learning）", "回归（Regression） 和 分类（Classification）有什么区别？")
    #     add("无监督学习（Unsupervised learning）", "我们如何通过非监督学习发现数据中的模式？")
    #     add("我们如何通过非监督学习发现数据中的模式？", "聚类分析（Clustering） 能帮助我们做什么？")
    #     add("我们如何通过非监督学习发现数据中的模式？", "K均值算法（K-means） 是聚类分析中常用的方法之一，它有什么优缺点？")
    #     add("我们如何通过非监督学习发现数据中的模式？", "异常检测（Anomaly Detection）在哪些方面有用？")
    #     add("我们如何通过非监督学习发现数据中的模式？", "降维（Dimensionality Reduction）有助于解决什么问题？")
    #     add("我们如何通过非监督学习发现数据中的模式？", "关联规则学习（Association Rule Learning）可以用于哪些场景？")
    #     add("强化学习（Reinforcement Learning）", "强化学习如何通过试错来让模型学习？")
    #     add("强化学习（Reinforcement Learning）", "Q 学习（Q-learning）的工作原理是什么？")
    #     add("聚类分析（Clustering）能帮助我们做什么？", "K 均值算法（K-means）是聚类分析中常用的方法之一，它有什么优缺点？")
    #     add("强化学习如何通过试错来让模型学习？", "我们可以使用哪些算法来实现强化学习？")
    #     add("强化学习如何通过试错来让模型学习？", "近端策略优化（Proximal Policy Optimization）是一种怎样的算法？")
    #     add("强化学习如何通过试错来让模型学习？", "Q 学习（Q-learning）的工作原理是什么？")
    # """

        self.parse_and_include_edges(output, replace=False)

    def parse_and_include_edges(self, output: str, replace: bool=True) -> None:
        """Parse output from LLM and include the edges in the graph.

        Args:
            output (str): output from LLM to be parsed
            replace (bool, optional): if True, replace all edges with the new ones, 
                otherwise add to existing edges. Defaults to True.
        """

        # Regex patterns
        pattern1 = r'(add|delete)\("([^()"]+)",\s*"([^()"]+)"\)'
        pattern2 = r'(delete)\("([^()"]+)"\)'

        matches1 = re.findall(pattern1, output)
        print(matches1)
        matches2 = re.findall(pattern2, output)
        print(matches2)


        # Find all matches in the text
        matches = re.findall(pattern1, output) + re.findall(pattern2, output)

        new_edges = []
        remove_edges = set()
        remove_nodes = set()

        for match in matches:
            op, *args = match
            add = op == "add"
            if add or (op == "delete" and len(args)==2):
                a, b = args
                if a == b:
                    continue
                if add:
                    new_edges.append((a, b))
                else:
                    # remove both directions
                    # (undirected graph)
                    remove_edges.add(frozenset([a, b]))
            else: # must be delete of node
                remove_nodes.add(args[0])

        if replace:
            edges = new_edges
        else:
            edges = self.edges + new_edges

        # make sure edges aren't added twice
        # and remove nodes/edges that were deleted
        added = set()
        for edge in edges:
            nodes = frozenset(edge)
            if nodes in added or nodes & remove_nodes or nodes in remove_edges:
                continue
            added.add(nodes)

        self.edges = list([tuple(a) for a in added])
        self.nodes = list(set([n for e in self.edges for n in e]))
        self.save()

    def _delete_node(self, node) -> None:
        """Delete a node and all edges connected to it.

        Args:
            node (str): The node to delete.
        """
        self.edges = [e for e in self.edges if node not in frozenset(e)]
        self.nodes = list(set([n for e in self.edges for n in e]))
        self.conversation.append(Message(
            f'delete("{node}")', 
            role="user"
        ))
        self.save()

    def _add_expand_delete_buttons(self, node) -> None:
        st.sidebar.subheader(node)
        cols = st.sidebar.columns(2)
        cols[0].button(
            label="扩展", 
            on_click=self.ask_for_extended_graph,
            key=f"expand_{node}",
            # pass to on_click (self.ask_for_extended_graph)
            kwargs={"selected_node": node}
        )
        cols[1].button(
            label="删除", 
            on_click=self._delete_node,
            type="primary",
            key=f"delete_{node}",
            # pass on to _delete_node
            args=(node,)
        )

    def visualize(self, graph_type: Literal["agraph", "networkx", "graphviz"]) -> None:
        """Visualize the mindmap as a graph a certain way depending on the `graph_type`.

        Args:
            graph_type (Literal["agraph", "networkx", "graphviz"]): The graph type to visualize the mindmap as.
        Returns:
            Union[str, None]: Any output from the clicking the graph or 
                if selecting a node in the sidebar.
        """

        selected = st.session_state.get("last_expanded")
        if graph_type == "agraph":
            vis_nodes = [
                Node(
                    id=n, 
                    label=n, 
                    # a little bit bigger if selected
                    size=10+10*(n==selected), 
                    # a different color if selected
                    color=COLOR if n != selected else FOCUS_COLOR
                ) 
                for n in self.nodes
            ]
            vis_edges = [Edge(source=a, target=b) for a, b in self.edges]
            config = Config(width="100%",
                            height=600,
                            directed=False, 
                            physics=True,
                            hierarchical=False,
                            )
            # returns a node if clicked, otherwise None
            clicked_node = agraph(nodes=vis_nodes, 
                            edges=vis_edges, 
                            config=config)
            # if clicked, update the sidebar with a button to create it
            if clicked_node is not None:
                self._add_expand_delete_buttons(clicked_node)
            return
        if graph_type == "networkx":
            graph = nx.Graph()
            for a, b in self.edges:
                graph.add_edge(a, b)
            colors = [FOCUS_COLOR if node == selected else COLOR for node in graph]
            fig, _ = plt.subplots(figsize=(16, 16))
            pos = nx.spring_layout(graph, seed = 123)
            nx.draw(graph, pos=pos, node_color=colors, with_labels=True)
            st.pyplot(fig)
        else: # graph_type == "graphviz":
            graph = graphviz.Graph()
            graph.attr(rankdir='LR')
            for a, b in self.edges:
                graph.edge(a, b, dir="both")
            for n in self.nodes:
                graph.node(n, style="filled", fillcolor=FOCUS_COLOR if n == selected else COLOR)
            #st.graphviz_chart(graph, use_container_width=True)
            b64 = base64.b64encode(graph.pipe(format='svg')).decode("utf-8")
            html = f"<img style='width: 100%' src='data:image/svg+xml;base64,{b64}'/>"
            st.write(html, unsafe_allow_html=True)
        # sort alphabetically
        for node in sorted(self.nodes):
            self._add_expand_delete_buttons(node)


def main():
    # will initialize the graph from session state
    # (if it exists) otherwise will create a new one
    mindmap = MindMap.load()

    st.sidebar.title("AI智能思维导航")

    # graph_type = st.sidebar.radio("图像类型", options=["agraph", "networkx", "graphviz"])
    graph_type = "agraph"
    empty = mindmap.is_empty()
    # reset = empty or st.sidebar.checkbox("重置", value=False)
    reset = True
    query = st.sidebar.text_area(
        "描述你的思维导图" if reset else "描述如何改变你的思维导图", 
        value=st.session_state.get("mindmap-input", ""),
        key="mindmap-input",
        height=200
    )
    print(query)
    # query = "机器学习"
    # mindmap.ask_for_initial_graph(query=query)
    submit = st.sidebar.button("提交")

    valid_submission = submit and query != ""

    if empty and not valid_submission:
        return

    with st.spinner(text="Loading graph..."):
        # if submit and non-empty query, then update graph
        if valid_submission:
            if reset:
                # completely new mindmap
                print( "\n# completely new mindmap:\n", query)
                mindmap.ask_for_initial_graph(query=query)
            else:
                # extend existing mindmap
                print( "\n# extend existing mindmap:\n", query)
                mindmap.ask_for_extended_graph(text=query)
            # since inputs also have to be updated, everything
            # is rerun
            # if st.st.rerun:
            #     st.write("The experimental_rerun function is available.")
            # else:
            #     st.write("The experimental_rerun function is not available.")
            # st.experimental_rerun()
            st.rerun()
        else:
            mindmap.visualize(graph_type)

if __name__ == "__main__":
    main()


# add("Socket Server和Client之间出现断连", "可能的原因")
# add("可能的原因", "网络问题")
# add("可能的原因", "服务器过载")
# add("可能的原因", "客户端程序异常")
# add("可能的原因", "服务器端程序异常")
# add("可能的原因", "配置错误")
# add("可能的原因", "防火墙或安全策略")
# add("网络问题", "路由器或交换机故障")
# add("网络问题", "DNS解析问题")
# add("网络问题", "带宽限制或拥堵")
# add("服务器过载", "资源分配不足")
# add("服务器过载", "并发连接数过多")
# add("客户端程序异常", "连接超时")
# add("客户端程序异常", "代码逻辑错误")
# add("服务器端程序异常", "连接处理错误")
# add("服务器端程序异常", "内存泄漏")
# add("配置错误", "IP地址或端口配置不正确")
# add("配置错误", "SSL/TLS配置错误")
# add("防火墙或安全策略", "阻止了合法的连接请求")
