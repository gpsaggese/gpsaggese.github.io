"""
NBA Data Analysis Utilities - Consolidated Functions

This module contains all functions from the NBA Analysis project consolidated into a single file.
All functions can be imported and used from a Jupyter notebook.
"""

import os
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
import json
import traceback
import gradio as gr

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# ============================================================================
# CONFIGURATION
# ============================================================================

# NBA Data Configuration
NBA_DATA_PATH = "nba24-25.csv"

# OpenAI Configuration (ONLY PROVIDER)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


def get_llm() -> LLM:
    """
    Create and return a CrewAI LLM instance configured for OpenAI.
    
    Returns:
        LLM: Configured CrewAI LLM instance for OpenAI
    
    Raises:
        ValueError: If OPENAI_API_KEY is not set
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it using: export OPENAI_API_KEY='your-api-key'"
        )
    return LLM(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY
    )


# ============================================================================
# VECTOR DATABASE
# ============================================================================

class NBAVectorDB:
    """
    Manages vector embeddings and semantic search for NBA data.
    Uses sentence-transformers for embeddings and ChromaDB for storage.
    """
    
    def __init__(self, csv_path: str, collection_name: str = "nba_data", persist_directory: str = "./chroma_db"):
        """
        Initialize the vector database.
        
        Args:
            csv_path: Path to the NBA CSV file
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector database
        """
        self.csv_path = csv_path
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize embedding model (open-source, runs locally)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded.")
        
        # Initialize ChromaDB client
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "NBA 2024-25 season data"}
        )
        
        # Check if collection is empty and needs indexing
        if self.collection.count() == 0:
            print("Vector database is empty. Indexing CSV data...")
            self._index_csv()
        else:
            print(f"Vector database loaded with {self.collection.count()} records")
    
    def _create_text_representation(self, row: pd.Series) -> str:
        """Convert a DataFrame row to a text representation for embedding."""
        parts = []
        
        if 'Player' in row:
            parts.append(f"Player: {row['Player']}")
        if 'Tm' in row:
            parts.append(f"Team: {row['Tm']}")
        if 'Opp' in row:
            parts.append(f"Opponent: {row['Opp']}")
        if 'Res' in row:
            parts.append(f"Result: {'Win' if row['Res'] == 'W' else 'Loss'}")
        if 'PTS' in row and pd.notna(row['PTS']):
            parts.append(f"Points: {row['PTS']}")
        if 'AST' in row and pd.notna(row['AST']):
            parts.append(f"Assists: {row['AST']}")
        if 'TRB' in row and pd.notna(row['TRB']):
            parts.append(f"Rebounds: {row['TRB']}")
        if 'FG%' in row and pd.notna(row['FG%']):
            parts.append(f"Field Goal Percentage: {row['FG%']:.3f}")
        if '3P%' in row and pd.notna(row['3P%']):
            parts.append(f"Three Point Percentage: {row['3P%']:.3f}")
        if 'Data' in row:
            parts.append(f"Date: {row['Data']}")
        
        return ". ".join(parts)
    
    def _index_csv(self):
        """Read CSV file, create embeddings, and store in ChromaDB."""
        print(f"Reading CSV from {self.csv_path}...")
        df = pd.read_csv(self.csv_path)
        
        print(f"Creating embeddings for {len(df)} records...")
        
        # Process in batches for efficiency
        batch_size = 100
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(df), batch_size):
            batch_df = df.iloc[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            batch_texts = []
            batch_metadatas = []
            batch_ids = []
            
            for idx, row in batch_df.iterrows():
                text = self._create_text_representation(row)
                batch_texts.append(text)
                
                metadata = {
                    'row_index': int(idx),
                    'player': str(row.get('Player', '')),
                    'team': str(row.get('Tm', '')),
                    'opponent': str(row.get('Opp', '')),
                    'result': str(row.get('Res', '')),
                    'points': float(row.get('PTS', 0)) if pd.notna(row.get('PTS')) else 0.0,
                    'date': str(row.get('Data', '')),
                }
                batch_metadatas.append(metadata)
                batch_ids.append(f"row_{idx}")
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} records)...")
            embeddings = self.embedding_model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()
            
            self.collection.add(
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        
        print(f"Indexed {len(df)} records in vector database.")
    
    def search(self, query: str, n_results: int = 10) -> List[Dict]:
        """
        Perform semantic search on the NBA data.
        
        Args:
            query: Natural language query
            n_results: Number of results to return
        
        Returns:
            List of dictionaries containing search results with metadata
        """
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True
        ).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity': 1 - results['distances'][0][i]
                })
        
        return formatted_results
    
    def get_original_row(self, row_index: int) -> Optional[pd.Series]:
        """Retrieve the original CSV row by index."""
        try:
            df = pd.read_csv(self.csv_path)
            if 0 <= row_index < len(df):
                return df.iloc[row_index]
        except Exception as e:
            print(f"Error retrieving row {row_index}: {e}")
        return None


# Global vector DB instance
_vector_db_instance: Optional[NBAVectorDB] = None


def get_vector_db(csv_path: str) -> NBAVectorDB:
    """
    Get or create the global vector database instance.
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        NBAVectorDB instance
    """
    global _vector_db_instance
    if _vector_db_instance is None or _vector_db_instance.csv_path != csv_path:
        _vector_db_instance = NBAVectorDB(csv_path)
    return _vector_db_instance


# ============================================================================
# TOOLS
# ============================================================================

def get_agent_tools(data_path: str):
    """
    Get the list of tools available for agents.
    
    Args:
        data_path: Path to the CSV data file
    
    Returns:
        list: List of tools for agents to use
    """
    
    def _read_nba_data(limit: int = 10) -> str:
        """Read a sample of the NBA data file to understand its structure."""
        try:
            df = pd.read_csv(data_path)
            sample = df.head(limit)
            return f"Dataset: {len(df):,} total records, {len(df.columns)} columns\n\nColumn names: {', '.join(df.columns.tolist())}\n\nSample (first {limit} rows):\n\n{sample.to_string()}"
        except Exception as e:
            return f"Error reading file {data_path}: {str(e)}"
    
    def _search_nba_data(
        query: Optional[str] = None,
        column: Optional[str] = None,
        value: Optional[str] = None,
        limit: int = 100
    ) -> str:
        """Search and filter NBA data CSV file."""
        try:
            df = pd.read_csv(data_path)
            
            if column and value:
                if column in df.columns:
                    df = df[df[column].astype(str).str.contains(str(value), case=False, na=False)]
                else:
                    return f"Column '{column}' not found. Available columns: {', '.join(df.columns.tolist())}"
            
            if query:
                mask = pd.Series([False] * len(df))
                for col in df.columns:
                    if df[col].dtype == 'object':
                        mask |= df[col].astype(str).str.contains(query, case=False, na=False)
                df = df[mask]
            
            limit = min(limit, 50)
            df = df.head(limit)
            
            if len(df) == 0:
                return "No matching records found."
            
            result_str = df.to_string()
            if len(result_str) > 2000:
                result_str = df.head(20).to_string() + f"\n\n... (showing first 20 of {len(df)} matching records)"
            
            return f"Found {len(df)} matching records:\n\n{result_str}"
        except Exception as e:
            return f"Error searching CSV {data_path}: {str(e)}"
    
    def _get_nba_data_summary() -> str:
        """Get a concise summary of the NBA data file."""
        try:
            df = pd.read_csv(data_path)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            summary = f"""NBA Dataset Summary:
- Total Records: {len(df):,}
- Columns: {len(df.columns)} ({', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''})
- Unique Players: {df['Player'].nunique() if 'Player' in df.columns else 'N/A'}
- Unique Teams: {df['Tm'].nunique() if 'Tm' in df.columns else 'N/A'}
- Date Range: {df['Data'].min() if 'Data' in df.columns else 'N/A'} to {df['Data'].max() if 'Data' in df.columns else 'N/A'}
- Key Numeric Columns: {', '.join(numeric_cols[:10]) if numeric_cols else 'None'}

Sample (first 3 rows):
{df.head(3).to_string()}
"""
            return summary
        except Exception as e:
            return f"Error getting CSV summary for {data_path}: {str(e)}"
    
    @tool("read_nba_data")
    def read_nba_data(limit: int = 10) -> str:
        """
        Read a sample of the NBA data file to understand its structure.
        
        Args:
            limit: Number of sample rows to return (default: 10, max: 50)
        """
        limit = min(limit, 50)
        return _read_nba_data(limit)
    
    @tool("search_nba_data")
    def search_nba_data(
        query: Optional[str] = None,
        column: Optional[str] = None,
        value: Optional[str] = None,
        limit: int = 100
    ) -> str:
        """
        Search and filter NBA data CSV file.
        
        Args:
            query: Optional text query to search for in any column
            column: Optional column name to filter by
            value: Optional value to match in the specified column
            limit: Maximum number of rows to return (default: 100)
        """
        return _search_nba_data(query, column, value, limit)
    
    @tool("get_nba_data_summary")
    def get_nba_data_summary() -> str:
        """Get a comprehensive summary of the NBA data file."""
        return _get_nba_data_summary()
    
    def _semantic_search_nba_data(query: str, n_results: int = 10) -> str:
        """Perform semantic search on NBA data using vector embeddings."""
        try:
            vector_db = get_vector_db(data_path)
            results = vector_db.search(query, n_results=n_results)
            
            if not results:
                return f"No results found for query: '{query}'"
            
            output = [f"Semantic search results for: '{query}'\n"]
            output.append(f"Found {len(results)} similar records:\n")
            output.append("=" * 80 + "\n")
            
            df = pd.read_csv(data_path)
            
            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                similarity = result['similarity']
                row_index = metadata.get('row_index', -1)
                
                output.append(f"\nResult {i} (Similarity: {similarity:.3f}):")
                output.append(f"Document: {result['document']}\n")
                
                if row_index >= 0 and row_index < len(df):
                    row = df.iloc[row_index]
                    output.append("Full record:")
                    output.append(row.to_string())
                    output.append("\n" + "-" * 80 + "\n")
            
            return "\n".join(output)
        except Exception as e:
            return f"Error performing semantic search: {str(e)}"
    
    @tool("semantic_search_nba_data")
    def semantic_search_nba_data(query: str, n_results: int = 10) -> str:
        """
        Perform semantic search on NBA data using vector embeddings.
        
        Args:
            query: Natural language query
            n_results: Number of results to return (default: 10, max: 50)
        """
        n_results = min(n_results, 50)
        return _semantic_search_nba_data(query, n_results)
    
    def _analyze_nba_data(pandas_code: str) -> str:
        """Execute pandas operations on NBA data for advanced analysis."""
        try:
            df = pd.read_csv(data_path)
            namespace = {
                'pd': pd,
                'df': df,
                '__builtins__': __builtins__
            }
            
            exec(f"result = {pandas_code}", namespace)
            result = namespace.get('result')
            
            if isinstance(result, pd.DataFrame):
                if len(result) > 50:
                    result_str = f"{result.head(50).to_string()}\n\n... (showing first 50 of {len(result)} rows)"
                else:
                    result_str = result.to_string()
                return f"Analysis Result ({result.shape[0]} rows, {result.shape[1]} cols):\n\n{result_str}"
            elif isinstance(result, pd.Series):
                if len(result) > 50:
                    result_str = f"{result.head(50).to_string()}\n\n... (showing first 50 of {len(result)} items)"
                else:
                    result_str = result.to_string()
                return f"Analysis Result ({len(result)} items):\n\n{result_str}"
            else:
                result_str = str(result)
                if len(result_str) > 2000:
                    result_str = result_str[:2000] + "\n\n... (truncated)"
                return f"Analysis Result:\n\n{result_str}"
                
        except Exception as e:
            return f"Error executing pandas code: {str(e)}\n\nMake sure your code uses 'df' as the DataFrame variable and returns a result."
    
    @tool("analyze_nba_data")
    def analyze_nba_data(pandas_code: str) -> str:
        """
        Execute pandas operations on NBA data for advanced analysis.
        
        Args:
            pandas_code: Valid pandas code that operates on a DataFrame variable named 'df'
        """
        return _analyze_nba_data(pandas_code)
    
    return [read_nba_data, search_nba_data, get_nba_data_summary, semantic_search_nba_data, analyze_nba_data]


# ============================================================================
# AGENTS
# ============================================================================

# Get LLM instance (shared across all agents)
_llm_instance = None

def _get_llm_instance():
    """Get or create shared LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = get_llm()
    return _llm_instance


def create_engineer_agent(csv_path: str = None) -> Agent:
    """
    Create the Engineer Agent for data processing and engineering tasks.
    
    Args:
        csv_path: Path to CSV file (defaults to NBA_DATA_PATH)
    
    Returns:
        Agent: Configured Engineer Agent
    """
    data_path = csv_path or NBA_DATA_PATH
    agent_tools = get_agent_tools(data_path)
    
    return Agent(
        role="Data Engineer",
        goal="Process, clean, and prepare data for analysis. Ensure data quality and create structured datasets.",
        backstory="""Data engineer specializing in sports analytics. 
        Responsible for processing datasets, handling missing values, data validation, 
        and creating clean, analysis-ready datasets.""",
        verbose=True,
        allow_delegation=False,
        llm=_get_llm_instance(),
        tools=agent_tools,
    )


def create_analyst_agent(csv_path: str = None) -> Agent:
    """
    Create the Analyst Agent for data analysis and insights.
    
    Args:
        csv_path: Path to CSV file (defaults to NBA_DATA_PATH)
    
    Returns:
        Agent: Configured Analyst Agent
    """
    data_path = csv_path or NBA_DATA_PATH
    agent_tools = get_agent_tools(data_path)
    
    return Agent(
        role="Data Analyst",
        goal="Analyze data to extract meaningful insights, identify patterns, and provide actionable recommendations.",
        backstory="""Data analyst focused on extracting insights. 
        Responsible for finding patterns in data, identifying trends, performing statistical analysis, 
        and translating data into clear insights.
        
        CRITICAL: When asked for aggregations, top N lists, totals, or statistical summaries:
        - ALWAYS use the 'analyze_nba_data' tool with pandas groupby operations
        - NEVER use semantic_search_nba_data for aggregation queries (it only returns individual records)
        - For "top 5 three-point shooters": use analyze_nba_data with groupby('Player')['3P'].sum()
        - Plan your analysis: understand what aggregation is needed, then write the appropriate pandas code""",
        verbose=True,
        allow_delegation=False,
        llm=_get_llm_instance(),
        tools=agent_tools,
    )


def create_storyteller_agent() -> Agent:
    """
    Create the Storyteller Agent for creating engaging headlines and storylines.
    
    Returns:
        Agent: Configured Storyteller Agent
    """
    return Agent(
        role="Sports Storyteller",
        goal="Transform data analysis results into engaging headlines and compelling storylines that bring statistics to life with narrative and context.",
        backstory="""Sports journalist creating headlines and storylines. 
        Responsible for turning statistical analysis into headlines and storylines. 
        Provides context and explains why the data matters.""",
        verbose=True,
        allow_delegation=False,
        llm=_get_llm_instance(),
        tools=[],  # Storyteller doesn't need data tools
    )


# ============================================================================
# TASKS
# ============================================================================

def create_data_engineering_task(engineer_agent, csv_path: str = None) -> Task:
    """
    Create the data engineering task for processing and cleaning data.
    
    Args:
        engineer_agent: The Engineer Agent to assign this task to
        csv_path: Path to CSV file (defaults to NBA_DATA_PATH)
    
    Returns:
        Task: Configured data engineering task
    """
    data_path = csv_path or NBA_DATA_PATH
    
    return Task(
        description=f"""
        Quickly examine the dataset located at {data_path}. 
        
        Your tasks (BE EFFICIENT - use tools only once):
        1. Get a brief summary of the dataset structure (use get_nba_data_summary ONCE)
        2. Note the key columns available
        3. Verify the data is ready for analysis
        
        IMPORTANT: 
        - Use get_nba_data_summary ONCE only - it provides all needed info
        - Do NOT call read_nba_data or analyze_nba_data multiple times
        - Keep your report concise (2-3 sentences)
        - The data is already clean and ready for analysis
        
        Provide a brief confirmation that the dataset is loaded and ready for analysis.
        """,
        agent=engineer_agent,
        expected_output="A brief confirmation (2-3 sentences) that the dataset is loaded and ready for analysis, including key column names."
    )


def create_data_analysis_task(analyst_agent, data_engineering_task: Task) -> Task:
    """
    Create the data analysis task for extracting insights from NBA data.
    
    Args:
        analyst_agent: The Analyst Agent to assign this task to
        data_engineering_task: The data engineering task for context
    
    Returns:
        Task: Configured data analysis task
    """
    return Task(
        description=f"""
        Using the cleaned NBA 2024-25 dataset, perform comprehensive analysis:
        
        Your tasks:
        1. Analyze player performance metrics:
           - Top performers by points, assists, rebounds
           - Shooting efficiency analysis (FG%, 3P%, FT%)
           - Player efficiency ratings
        2. Team performance analysis:
           - Win/loss records by team
           - Team offensive and defensive statistics
           - Team performance trends
        3. Game insights:
           - High-scoring games
           - Close games vs blowouts
           - Performance by date/period
        4. Identify key patterns and trends:
           - Best performing players
           - Most efficient teams
           - Statistical outliers
        5. Provide actionable insights and recommendations
        
        Create a comprehensive analysis report with key findings and insights.
        """,
        agent=analyst_agent,
        expected_output="A detailed analysis report with key insights, statistical findings, top performers, team analysis, and actionable recommendations based on the NBA 2024-25 data.",
        context=[data_engineering_task]
    )


def create_custom_analysis_task(analyst_agent, user_query: str, data_engineering_task: Task = None, csv_path: str = None) -> Task:
    """
    Create a custom data analysis task based on user input.
    
    Args:
        analyst_agent: The Analyst Agent to assign this task to
        user_query: The user's custom analysis query/task
        data_engineering_task: The data engineering task for context (optional for parallel execution)
        csv_path: Path to CSV file (for reference in description)
    
    Returns:
        Task: Configured custom analysis task
    """
    data_path = csv_path or NBA_DATA_PATH
    context = [data_engineering_task] if data_engineering_task else []
    
    return Task(
        description=f"""
        Using the dataset located at {data_path}, perform the following analysis as requested by the user:
        
        {user_query}
        
        IMPORTANT INSTRUCTIONS:
        1. For queries requiring aggregations (sum, count, average, top N, etc.), you MUST use the 'analyze_nba_data' tool.
        2. The 'analyze_nba_data' tool allows you to execute pandas code for grouping, aggregating, sorting, and filtering.
        3. Examples of when to use 'analyze_nba_data':
           - Finding top players by statistics (e.g., "top 5 three-point shooters")
           - Calculating totals or averages per player/team
           - Grouping and aggregating data
           - Statistical analysis requiring groupby operations
        4. Use 'semantic_search_nba_data' only for finding specific game records or examples, NOT for aggregations.
        5. Plan your analysis: First understand what data you need, then use the appropriate tool to get aggregated results.
        
        Steps to follow:
        1. If the query asks for "top N" or aggregations, use analyze_nba_data with pandas groupby operations
        2. For "top 5 three-point shooters": group by Player, sum the '3P' column, sort descending, take top 5
        3. Present the results clearly with player names and their statistics
        
        Provide a clear, comprehensive answer with relevant statistics, insights, and any supporting data from the dataset.
        """,
        agent=analyst_agent,
        expected_output="A detailed analysis report addressing the user's query with relevant insights, statistics, and findings from the data.",
        context=context
    )


def create_storyteller_task(storyteller_agent, analysis_task: Task) -> Task:
    """
    Create a storyteller task that creates headlines and storylines from the analysis results.
    
    Args:
        storyteller_agent: The Storyteller Agent to assign this task to
        analysis_task: The analysis task whose output will be used to create headlines and content
    
    Returns:
        Task: Configured storyteller task
    """
    return Task(
        description="""
        Review the data analysis results and create engaging headlines and compelling storylines that bring the data to life.
        
        Your tasks:
        1. Read and understand the analysis results thoroughly
        2. Identify the most important and interesting findings
        3. Create 3-5 compelling headlines that:
           - Are catchy and attention-grabbing
           - Accurately reflect the key insights
           - Use engaging sports journalism language
           - Are suitable for display to users
        
        4. Write engaging storylines/content for each headline that:
           - Tells a story about the findings
           - Provides context and narrative around the statistics
           - Makes the data come alive with compelling prose
           - Explains why these insights matter
           - Uses vivid language and storytelling techniques
           - Is 2-3 paragraphs per storyline (enough to be engaging but concise)
        
        5. Format your output as follows:
           HEADLINES:
           [List of 3-5 headlines, one per line]
           
           STORYLINES:
           [For each headline, write 2-3 paragraphs of engaging content that tells the story behind the data]
        
        Make both the headlines and storylines exciting, memorable, and true to the data insights. 
        Write like a sports journalist who knows how to make statistics compelling and human.
        """,
        agent=storyteller_agent,
        expected_output="A formatted output with 3-5 engaging headlines followed by detailed storylines (2-3 paragraphs each) that bring the data analysis to life with compelling narrative and context.",
        context=[analysis_task]
    )


# ============================================================================
# CREW CREATION
# ============================================================================

def create_crew() -> Crew:
    """
    Create and configure the CrewAI crew with agents and tasks.
    
    Returns:
        Crew: Configured CrewAI crew ready for execution
    """
    engineer_agent = create_engineer_agent()
    analyst_agent = create_analyst_agent()
    
    data_engineering_task = create_data_engineering_task(engineer_agent)
    data_analysis_task = create_data_analysis_task(analyst_agent, data_engineering_task)
    
    return Crew(
        agents=[engineer_agent, analyst_agent],
        tasks=[data_engineering_task, data_analysis_task],
        process=Process.sequential,
        verbose=True,
    )


def create_crew_with_custom_task(user_query: str, csv_path: str = None) -> Crew:
    """
    Create a CrewAI crew with engineering task, custom analyst task, and storyteller task.
    
    Args:
        user_query: The user's custom analysis query/task
        csv_path: Optional path to CSV file (if None, uses default from config)
    
    Returns:
        Crew: Configured CrewAI crew ready for execution
    """
    engineer_agent = create_engineer_agent(csv_path)
    analyst_agent = create_analyst_agent(csv_path)
    storyteller_agent = create_storyteller_agent()
    
    data_engineering_task = create_data_engineering_task(engineer_agent, csv_path)
    custom_analysis_task = create_custom_analysis_task(analyst_agent, user_query, None, csv_path)
    storyteller_task = create_storyteller_task(storyteller_agent, custom_analysis_task)
    
    return Crew(
        agents=[engineer_agent, analyst_agent, storyteller_agent],
        tasks=[data_engineering_task, custom_analysis_task, storyteller_task],
        process=Process.sequential,
        verbose=True,
    )


def create_flow_crew(user_query: str, csv_path: str) -> Crew:
    """
    Create a single crew with parallel tasks (Engineer and Analyst) that merge results at the end.
    
    Args:
        user_query: The user's custom analysis query/task
        csv_path: Path to the uploaded CSV file
    
    Returns:
        Crew: Single crew with parallel tasks that will merge results
    """
    engineer_agent = create_engineer_agent(csv_path)
    analyst_agent = create_analyst_agent(csv_path)
    storyteller_agent = create_storyteller_agent()
    
    data_engineering_task = create_data_engineering_task(engineer_agent, csv_path)
    custom_analysis_task = create_custom_analysis_task(analyst_agent, user_query, None, csv_path)
    storyteller_task = create_storyteller_task(storyteller_agent, custom_analysis_task)
    
    return Crew(
        agents=[engineer_agent, analyst_agent, storyteller_agent],
        tasks=[data_engineering_task, custom_analysis_task, storyteller_task],
        process=Process.sequential,
        verbose=True,
    )


def create_analysis_only_crew(user_query: str, csv_path: str) -> Crew:
    """
    Create a crew with only Analyst and Storyteller agents (no Engineer).
    
    Args:
        user_query: The user's custom analysis query/task
        csv_path: Path to the uploaded CSV file
    
    Returns:
        Crew: Crew with only analyst and storyteller tasks
    """
    analyst_agent = create_analyst_agent(csv_path)
    storyteller_agent = create_storyteller_agent()
    
    custom_analysis_task = create_custom_analysis_task(analyst_agent, user_query, None, csv_path)
    storyteller_task = create_storyteller_task(storyteller_agent, custom_analysis_task)
    
    return Crew(
        agents=[analyst_agent, storyteller_agent],
        tasks=[custom_analysis_task, storyteller_task],
        process=Process.sequential,
        verbose=True,
    )


def create_analyst_only_crew(user_query: str, csv_path: str) -> Crew:
    """
    Create a crew with only Analyst agent (no Engineer, no Storyteller).
    
    Args:
        user_query: The user's custom analysis query/task
        csv_path: Path to the uploaded CSV file
    
    Returns:
        Crew: Crew with only analyst task
    """
    analyst_agent = create_analyst_agent(csv_path)
    custom_analysis_task = create_custom_analysis_task(analyst_agent, user_query, None, csv_path)
    
    return Crew(
        agents=[analyst_agent],
        tasks=[custom_analysis_task],
        process=Process.sequential,
        verbose=True,
    )


# ============================================================================
# APP FUNCTIONS
# ============================================================================

def process_file_and_analyze(file, user_query: str = "", engineer_result: str = None) -> Tuple[str, str]:
    """
    Process uploaded file and run all agents (Engineer, Analyst, Storyteller), then merge results.
    
    Args:
        file: Uploaded file object
        user_query: The user's analysis query/task (empty for general analysis)
        engineer_result: Previously computed engineer result (if available)
    
    Returns:
        tuple: (merged_results, engineer_result) - engineer_result is stored for reuse
    """
    if file is None:
        return "Please upload a CSV file.", engineer_result or ""
    
    if not user_query or not user_query.strip():
        user_query = "Provide a comprehensive analysis of the dataset including: top performers, key statistics, interesting patterns, and notable insights."
    
    try:
        file_path = file.name if hasattr(file, 'name') else str(file)
        csv_path = file_path
        
        crew = create_flow_crew(user_query.strip(), csv_path)
        result = crew.kickoff()
        
        merged_output = []
        stored_engineer_result = ""
        
        if hasattr(result, 'tasks_output') and result.tasks_output:
            if len(result.tasks_output) >= 1:
                engineer_output = str(result.tasks_output[0])
                stored_engineer_result = engineer_output
                merged_output.append("## Engineer Agent Results")
                merged_output.append("")
                merged_output.append(engineer_output)
                merged_output.append("")
                merged_output.append("---")
                merged_output.append("")
        
        if hasattr(result, 'tasks_output') and result.tasks_output:
            if len(result.tasks_output) >= 2:
                analyst_output = str(result.tasks_output[1])
                merged_output.append("## Analyst Agent Results")
                merged_output.append("")
                merged_output.append(analyst_output)
                merged_output.append("")
                merged_output.append("---")
                merged_output.append("")
        
        if hasattr(result, 'tasks_output') and result.tasks_output:
            if len(result.tasks_output) >= 3:
                storyteller_output = str(result.tasks_output[2])
                merged_output.append("## Storyteller Agent Results")
                merged_output.append("")
                merged_output.append(storyteller_output)
                merged_output.append("")
        
        if not merged_output:
            merged_output.append("## Complete Analysis Results")
            merged_output.append("")
            merged_output.append(str(result))
        
        return "\n".join(merged_output), stored_engineer_result
    
    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{error_trace}"
        print(error_msg)
        return error_msg, engineer_result or ""


def process_question_only(file, user_query: str) -> str:
    """
    Process a specific user question using only the Analyst agent.
    
    Args:
        file: Uploaded file object
        user_query: The user's specific analysis question
    
    Returns:
        str: Analyst results only
    """
    if file is None:
        return "Please upload a CSV file."
    
    if not user_query or not user_query.strip():
        return "Please enter a question."
    
    try:
        file_path = file.name if hasattr(file, 'name') else str(file)
        csv_path = file_path
        
        crew = create_analyst_only_crew(user_query.strip(), csv_path)
        result = crew.kickoff()
        
        if hasattr(result, 'tasks_output') and result.tasks_output:
            if len(result.tasks_output) >= 1:
                analyst_output = str(result.tasks_output[0])
                return analyst_output
        
        return str(result)
    
    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{error_trace}"
        print(error_msg)
        return error_msg


def create_app():
    """Create and return the Gradio interface."""
    with gr.Blocks(title="NBA Stats Analysis with CrewAI", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # NBA Stats Analysis with CrewAI
        
        Upload your NBA statistics CSV file to get comprehensive analysis with engaging storylines.
        
        **How it works:**
        - **Engineer Agent**: Examines and validates your dataset
        - **Analyst Agent**: Performs deep analysis (general or based on your question)
        - **Storyteller Agent**: Creates headlines and compelling storylines
        
        All agents work in parallel and results are merged for you!
        """)
        
        engineer_state = gr.State(value="")
        
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload CSV File",
                    file_types=[".csv"],
                    type="filepath"
                )
                
                analyze_btn = gr.Button(
                    "Analyze Dataset", 
                    variant="primary", 
                    size="lg",
                    visible=False
                )
                
                gr.Markdown("### Ask a Specific Question")
                
                query_input = gr.Textbox(
                    label="Your Analysis Question",
                    placeholder="e.g., 'Who are the top 5 three-point shooters?' or 'Analyze the best players by assists'",
                    lines=2
                )
                
                question_output = gr.Markdown(
                    value="",
                    label="Answer",
                    visible=False
                )
                
                query_btn = gr.Button(
                    "Analyze with Question", 
                    variant="secondary", 
                    size="lg"
                )
        
        with gr.Row():
            with gr.Column():
                status_output = gr.Markdown(
                    value="",
                    label="Agent Status",
                    visible=False
                )
        
        with gr.Row():
            with gr.Column():
                merged_output = gr.Markdown(
                    value="**Ready to analyze!** Upload a CSV file above, then click 'Analyze Dataset' to get started.",
                    label="Full Analysis Results"
                )
        
        def show_loading_animation(is_question: bool = False):
            """Show loading animation while processing."""
            if is_question:
                return """## Analysis in Progress...

<div style="text-align: center; padding: 20px;">
    <div style="font-size: 18px; margin-bottom: 15px;">
        <strong>Analyzing your question...</strong>
    </div>
    <div style="display: flex; justify-content: center; max-width: 600px; margin: 0 auto;">
        <div style="text-align: center; margin: 10px;">
            <div style="font-size: 14px; font-weight: bold;">Analyst Agent</div>
            <div style="font-size: 12px; color: #666; margin-top: 5px;">Processing query...</div>
        </div>
    </div>
    <div style="margin-top: 25px; font-size: 14px; color: #888;">
        This may take a moment... Please wait while the agent processes your question.
    </div>
</div>"""
            else:
                return """## Analysis in Progress...

<div style="text-align: center; padding: 20px;">
    <div style="font-size: 18px; margin-bottom: 15px;">
        <strong>Agents are working in parallel...</strong>
    </div>
    <div style="display: flex; justify-content: space-around; max-width: 600px; margin: 0 auto; flex-wrap: wrap;">
        <div style="text-align: center; margin: 10px;">
            <div style="font-size: 14px; font-weight: bold;">Engineer Agent</div>
            <div style="font-size: 12px; color: #666; margin-top: 5px;">Examining dataset...</div>
        </div>
        <div style="text-align: center; margin: 10px;">
            <div style="font-size: 14px; font-weight: bold;">Analyst Agent</div>
            <div style="font-size: 12px; color: #666; margin-top: 5px;">Analyzing data...</div>
        </div>
        <div style="text-align: center; margin: 10px;">
            <div style="font-size: 14px; font-weight: bold;">Storyteller Agent</div>
            <div style="font-size: 12px; color: #666; margin-top: 5px;">Creating storylines...</div>
        </div>
    </div>
    <div style="margin-top: 25px; font-size: 14px; color: #888;">
        This may take a moment... Please wait while the agents process your data.
    </div>
</div>"""
        
        def on_file_upload(file):
            """Handle file upload - show analyze button and reset state."""
            if file is not None:
                return gr.update(visible=True), ""
            return gr.update(visible=False), ""
        
        def start_full_analysis(file, engineer_result: str = ""):
            """Start full analysis and show loading animation."""
            loading_msg = show_loading_animation(is_question=False)
            return gr.update(visible=True, value=loading_msg), gr.update(value="")
        
        def complete_full_analysis(file, engineer_result: str = ""):
            """Complete full analysis and return results."""
            result, new_engineer_result = process_file_and_analyze(file, "", engineer_result)
            if result.startswith("Error:") or result.startswith("Please upload"):
                result = f"### {result}"
            return result, gr.update(visible=False), new_engineer_result
        
        def start_question_analysis(file, user_query: str = ""):
            """Start question analysis and show loading animation."""
            loading_msg = show_loading_animation(is_question=True)
            return gr.update(visible=True, value=loading_msg), gr.update(visible=True, value="")
        
        def complete_question_analysis(file, user_query: str = ""):
            """Complete question analysis and return results."""
            result = process_question_only(file, user_query)
            if result.startswith("Error:") or result.startswith("Please"):
                result = f"### {result}"
            else:
                result = f"""<div style="background-color: #f0f7ff; border: 2px solid #4a90e2; border-radius: 8px; padding: 15px; margin: 10px 0;">
{result}
</div>"""
            return result, gr.update(visible=False)
        
        file_input.change(
            fn=on_file_upload,
            inputs=[file_input],
            outputs=[analyze_btn, engineer_state]
        )
        
        analyze_btn.click(
            fn=start_full_analysis,
            inputs=[file_input, engineer_state],
            outputs=[status_output, merged_output]
        ).then(
            fn=complete_full_analysis,
            inputs=[file_input, engineer_state],
            outputs=[merged_output, status_output, engineer_state]
        )
        
        query_btn.click(
            fn=start_question_analysis,
            inputs=[file_input, query_input],
            outputs=[status_output, question_output]
        ).then(
            fn=complete_question_analysis,
            inputs=[file_input, query_input],
            outputs=[question_output, status_output]
        )
        
        query_input.submit(
            fn=start_question_analysis,
            inputs=[file_input, query_input],
            outputs=[status_output, question_output]
        ).then(
            fn=complete_question_analysis,
            inputs=[file_input, query_input],
            outputs=[question_output, status_output]
        )
    
    return app


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run the NBA data analysis crew."""
    print("=" * 60)
    print("NBA 2024-25 Data Analysis with CrewAI")
    print("Using LLM Provider: OPENAI")
    print("=" * 60)
    print()
    
    if not os.path.exists(NBA_DATA_PATH):
        print(f"Error: {NBA_DATA_PATH} not found!")
        return
    
    print(f"Loading data from {NBA_DATA_PATH}...")
    try:
        df = pd.read_csv(NBA_DATA_PATH)
        print(f"Dataset loaded: {len(df)} records, {len(df.columns)} columns")
        print(f"Columns: {', '.join(df.columns.tolist())}")
        print()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print("Starting CrewAI agents...")
    print("Engineer Agent will process and clean the data...")
    print("Analyst Agent will analyze the data for insights...")
    print()
    print("-" * 60)
    print()
    
    crew = create_crew()
    result = crew.kickoff()
    
    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print()
    print(result)

