import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import json
import random
from pyvis.network import Network
from openai import OpenAI
from dotenv import load_dotenv

def initialize_df() -> pd.DataFrame:
    df = pd.read_csv('faculty_data_final.csv', encoding='ISO-8859-1')

    df = df.rename(
            columns={
                'Name': 'name',
                'School': "school",
                'Degree Program': "degree_program",
                'UGA Affiliations (e.g. Centers or Institutes etc.)': "uga_affiliations",
                'Previous Instutution(s)': "previous_institutions",
                'PhD. Degree': "phd_degree",
                'Interdisciplinary Areas': "interdisciplinary_areas",
                'Broad Speacialty Areas / Expertise': "broad_specialties",
                'Overlapping Expertise': "overlapping_expertise",
                'Research Keywords': "research_keywords",
                'Major Tool / Equipment': "equipment",
                'Potential Sponsors': "potential_sponsors",
                'UGA Collaborator(s)': "uga_collaborators",
                'Outside Collaborator(s)': "outside_collaborators",
                'Global Engagement': "global_engagement",
                'Memberships': "memberships",
                'Other Information': "other"
            }
        )

    df['name'] = df['name'].ffill()
    df['name'] = df['name'].str.strip()
    df = df.groupby('name').agg(lambda x: ', '.join(x.dropna().astype(str)))

    def combine_entries(series):
        if series.dtype == 'object':
            return ', '.join(set(', '.join(series.dropna()).split(', ')))
        else:
            return series.dropna().iloc[0]

    df = df.groupby('name').agg(combine_entries).reset_index()

    return df

def create_columns(df) -> pd.DataFrame:
    gen_df = df[["name", "interdisciplinary_areas", "broad_specialties", "outside_collaborators"]]
    json_data = gen_df.to_json(orient="records", indent=4)

    API_KEY = os.getenv("API_KEY")

    client = OpenAI(
        api_key=API_KEY
    )

    prompt = f"""
        You are categorizing engineering professors at the University of Georgia based on their interdisciplinary research areas and broad specialties.

        ## **Task Requirements**
        1. **You must group all professors into exactly 4-5 broad categories.**  
           - Example categories:  
               - "AI, Data Science, and Cyber-Physical Systems"  
               - "Biomedical and Health Engineering"  
               - "Energy, Environment, and Sustainability"  
               - "Materials, Manufacturing, and Robotics"  
               - "Education, Policy, and Social Impact in Engineering"  
           - **Categories must remain broad and standardized** while maintaining meaningful research connections.  
           - You may **slightly adjust** the names but **must not exceed 5 groups**.  

        2. **Strict Assignment Rules:**  
           - **Every professor must be assigned to one and only one category.**  
           - If a professor's research spans multiple areas, **assign them to the closest matching category.**  
           - If a professor's research data is completely empty, assign them `""` (empty string).  

        3. **Category Balance Enforcement:**  
           - If a category has **only one professor**, merge it with the most related category.  
           - If a category has an excessive number of professors, **split only if absolutely necessary** (and still ensure a max of 5 groups).  
           - The **final output must always contain 4-5 categories**, no more, no less.  

        4. **STRICT JSON Output Format:**  
           Your response **MUST BE STRICTLY JSON** and match the format below:  
        ```json
        {{
            "insight": "Some professors had highly interdisciplinary research, requiring careful classification.",
            "generated_disciplines": [
                {{
                    "name": "Professor A",
                    "discipline": "AI, Data Science, and Cyber-Physical Systems"
                }},
                {{
                    "name": "Professor B",
                    "discipline": "Biomedical and Health Engineering"
                }},
                {{
                    "name": "Professor C",
                    "discipline": "Materials, Manufacturing, and Robotics"
                }},
                ...
            ]
        }}
        ```

        **Deviating from this format will be considered incorrect output.**  

        ## **Input JSON Data**
        ```json
        {json_data}
        ```
    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=[{
            "role": "user", 
            "content": prompt
        }]
    )

    raw_content = completion.choices[0].message.content.strip()

    if raw_content.startswith("```json"):
        raw_content = raw_content[7:-3]

    data = json.loads(raw_content)
    disciplines_df = pd.DataFrame(data["generated_disciplines"])
    disciplines_df.set_index("name", inplace=True)
    disciplines_df.index = disciplines_df.index.str.strip()
    
    df_json = df.to_json(orient="records", indent=4)
    n"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=[{
            "role": "user", 
            "content": second_prompt
        }]
    )
    raw_content = completion.choices[0].message.content.strip()

    if raw_content.startswith("```json"):
        raw_content = raw_content[7:-3]

    relatedness_data = json.loads(raw_content)

    relatedness_df = pd.DataFrame(relatedness_data["generated_groups"])
    relatedness_df = relatedness_df.reset_index(drop=True).set_index('name')

    # Convert to tuples
    relatedness_df["related_professors"] = relatedness_df["related_professors"].apply(
        lambda x: [(rel["name"], float(rel["weight"])) for rel in x]
    )

    relatedness_df.index = relatedness_df.index.str.strip()

    df = df.reset_index(drop=True).set_index('name')
    df = df.merge(disciplines_df, left_index=True, right_index=True, how="left")
    df = df.merge(relatedness_df, left_index=True, right_index=True, how="left")

    return df



def plot_graph(df, search_professor=False, category_search_parameter="degree_program", professor_search_parameter=None, min_weight=0):
    if not search_professor:
        graphs = {}
        category_counts = df[category_search_parameter].value_counts()
        single_nodes = category_counts[category_counts == 1].index
        miscellaneous = []

        for value in df[category_search_parameter].unique():
            if pd.isna(value) or value == "":
                continue

            subset = df[df[category_search_parameter] == value]

            if value in single_nodes:
                miscellaneous.append(subset.index[0])
                continue

            net = Network(notebook=False)
            nodes = list(subset.index)

            for node in nodes:
                net.add_node(node, label=str(node))

            for i in range(1, len(nodes)):
                net.add_edge(nodes[i - 1], nodes[i])

            if len(nodes) > 3:
                cycle_start = random.choice(nodes)
                cycle_end = random.choice(nodes)
                net.add_edge(cycle_start, cycle_end)

            if len(nodes) > 4:
                center_node = nodes[0]
                for node in nodes[1:]:
                    net.add_edge(center_node, node)

            graphs[value] = net

        if miscellaneous:
            net_misc = Network(notebook=False)

            for node in miscellaneous:
                net_misc.add_node(node, label=str(node))

            for i in range(1, len(miscellaneous)):
                net_misc.add_edge(miscellaneous[i - 1], miscellaneous[i])

            if len(miscellaneous) > 3:
                cycle_start = random.choice(miscellaneous)
                cycle_end = random.choice(miscellaneous)
                net_misc.add_edge(cycle_start, cycle_end)

            if len(miscellaneous) > 4:
                center_node = miscellaneous[0]
                for node in miscellaneous[1:]:
                    net_misc.add_edge(center_node, node)

            graphs["Miscellaneous"] = net_misc

        for value, net in graphs.items():
            net.show(f"graph_{value}.html")

    elif search_professor:
        professor_data = df[df.index == professor_search_parameter]

        if not professor_data.empty:
            related_professors = professor_data['related_professors'].values[0]
            net = Network(notebook=False)
            net.add_node(professor_search_parameter, label=str(professor_search_parameter), color='red')
            added_nodes = {professor_search_parameter}
            visited = set([professor_search_parameter])
            layer_queue = [(professor_search_parameter, related_professors)]

            while layer_queue:
                current_layer = layer_queue.pop(0)
                current_professor, related_professors = current_layer

                for related_professor, weight in related_professors:
                    if min_weight > weight:
                        continue
                    if related_professor not in added_nodes:
                        net.add_node(related_professor, label=str(related_professor), color='lightgreen')
                        net.add_edge(current_professor, related_professor, title=str(weight))
                        added_nodes.add(related_professor)
                        visited.add(related_professor)

                        if related_professor in df.index:
                            next_layer_professors = df.loc[related_professor, 'related_professors']
                            if isinstance(next_layer_professors, float):
                                continue
                            for next_related_professor, _ in next_layer_professors:
                                if next_related_professor not in visited:
                                    layer_queue.append((related_professor, next_layer_professors))

            net.show(f"graph_{professor_search_parameter}.html")
        else:
            print(f"Professor '{professor_search_parameter}' not found in the dataset.")

"""
def plot_graph(df, 
               search_professor=False, 
               category_search_parameter="degree_program", 
               professor_search_parameter=None,
               min_weight=0
    ):
    if not search_professor:
        graphs = {}

        category_counts = df[category_search_parameter].value_counts()
        single_nodes = category_counts[category_counts == 1].index

        # Place items with no connections in miscellaneous
        miscellaneous = []

        for value in df[category_search_parameter].unique():
            if pd.isna(value) or value == "":
                continue

            subset = df[df[category_search_parameter] == value]

            if value in single_nodes:
                miscellaneous.append(subset.index[0])
                continue

            G = nx.Graph()
            nodes = list(subset.index)
            G.add_nodes_from(nodes)

            # Create a tree (connected graph with no cycles) by adding edges sequentially
            for i in range(1, len(nodes)):
                G.add_edge(nodes[i - 1], nodes[i])

            if len(nodes) > 3:  # Add a cycle for larger graphs
                cycle_start = random.choice(nodes)
                cycle_end = random.choice(nodes)
                G.add_edge(cycle_start, cycle_end)  # Add a cycle by connecting two random nodes

            if len(nodes) > 4:
                center_node = nodes[0]
                for node in nodes[1:]:
                    G.add_edge(center_node, node)

            graphs[value] = G

        # If there are any "Miscellaneous" nodes, create a graph for them
        if miscellaneous:
            G_misc = nx.Graph()
            G_misc.add_nodes_from(miscellaneous)
            
            for i in range(1, len(miscellaneous)):
                G_misc.add_edge(miscellaneous[i - 1], miscellaneous[i])

            if len(miscellaneous) > 3:
                cycle_start = random.choice(miscellaneous)
                cycle_end = random.choice(miscellaneous)
                G_misc.add_edge(cycle_start, cycle_end)

            if len(miscellaneous) > 4:
                center_node = miscellaneous[0]
                for node in miscellaneous[1:]:
                    G_misc.add_edge(center_node, node)

            graphs["Miscellaneous"] = G_misc

        # Visualize the graphs
        num_graphs = len(graphs)
        cols = 2
        rows = (num_graphs + cols - 1) // cols

        plt.figure(figsize=(cols * 5, rows * 4))

        for i, (value, G) in enumerate(graphs.items(), 1):
            plt.subplot(rows, cols, i)            
            
            pos = nx.spring_layout(G, k=0.2, seed=42)
            
            nx.draw(
                G, pos, with_labels=True, node_color='lightblue', node_size=1500,
                font_size=10, font_weight='normal', edge_color='gray', width=2,
                alpha=0.7, edgecolors="black"
            )

            plt.title(value, fontsize=12)

        plt.tight_layout()
        plt.show()

    elif search_professor:
        professor_data = df[df.index == professor_search_parameter]

        if not professor_data.empty:
            related_professors = professor_data['related_professors'].values[0]
            
            G = nx.Graph()
            G.add_node(professor_search_parameter)
            added_nodes = {professor_search_parameter}
            visited = set([professor_search_parameter])
            
            # Perform BFS for 1st, 2nd, and further layers
            layer_queue = [(professor_search_parameter, related_professors)]  # (professor, related_professors)
            
            # For each layer, we'll expand the related professors
            while layer_queue:
                current_layer = layer_queue.pop(0)  # Get the next professor and their related professors
                current_professor, related_professors = current_layer
                
                # Add all related professors from this layer
                for related_professor, weight in related_professors:
                    if min_weight > weight:
                        continue
                    if related_professor not in added_nodes:
                        G.add_node(related_professor)
                        G.add_edge(current_professor, related_professor, weight=weight)
                        added_nodes.add(related_professor)
                        visited.add(related_professor)  # Mark as visited

                        # Add all related professors of the current professor to the queue for the next layer
                        if related_professor in df.index:
                            next_layer_professors = df.loc[related_professor, 'related_professors']

                            if isinstance(next_layer_professors, float):
                                continue

                            # Only add to the queue if this professor hasn't been visited already
                            for next_related_professor, _ in next_layer_professors:                            
                                if next_related_professor not in visited:
                                    layer_queue.append((related_professor, next_layer_professors))

            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G, k=0.2, seed=42)

            node_colors = ["pink" if node == professor_search_parameter else "lightgreen" for node in G.nodes]

            nx.draw(
                G, pos, with_labels=True, node_color=node_colors, node_size=1500,
                font_size=12, font_weight='normal', edge_color='gray', width=2,
                alpha=0.7, edgecolors="black"
            )
            
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

            plt.title(f"Professor: {professor_search_parameter} and Related Professors", fontsize=14)
            
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            plt.show()

        else:
            print(f"Professor '{professor_search_parameter}' not found in the dataset.")
"""
if __name__ == "__main__":
    load_dotenv()
    df = initialize_df()
    df = create_columns(df)
    plot_graph(
        df, 
        search_professor=True,
        professor_search_parameter="Christina Fuller"
    )
