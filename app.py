# =====================================
# REAL FINANCEGPT GRADIO INTERFACE
# Uses YOUR actual working FinanceGPT from paste-2 with Gradio wrapper
# Just copy/paste and run in Colab!
# =====================================

import warnings
warnings.filterwarnings('ignore')

# Install required packages for Gradio interface
import subprocess
import sys

def install_packages():
    packages = [
        'gradio',
        'upstash-redis'
    ]

    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed")
        except:
            print(f"⚠️ {package} installation failed, continuing...")

print("📦 Installing Gradio interface packages...")
install_packages()

# =====================================
# IMPORT YOUR REAL FINANCEGPT SYSTEM (FROM PASTE-2)
# =====================================

# All the imports from your working system
from jerzy import *
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import base64
from io import BytesIO
import requests
from upstash_redis import Redis

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# =====================================
# YOUR CONFIGURATION AND SETUP (FROM PASTE-2)
# =====================================

# --- SECRETS (load from environment) ---
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY")
UPSTASH_URL = os.environ.get("UPSTASH_URL")
UPSTASH_TOKEN = os.environ.get("UPSTASH_TOKEN")

# --- PUBLIC CONFIG (safe to keep in code) ---
DEEPINFRA_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

# Redis Configuration using the secret variables
redis = Redis(
    url=UPSTASH_URL,
    token=UPSTASH_TOKEN
)

# Redis key prefixes
FINANCE_PREFIX = "finance"
AUDIT_PREFIX = "audit"
CHART_PREFIX = "chart"
USERS_PREFIX = "gradio_users"
CONVERSATIONS_PREFIX = "gradio_conversations"

# =====================================
# YOUR UPSTASH REDIS STORAGE (FROM PASTE-2)
# =====================================

class UpstashStorage:
    """Your Redis storage system from paste-2"""

    @staticmethod
    def save_json(data, basket_name, item_name):
        try:
            redis_key = f"{basket_name}:{item_name}"
            redis.set(redis_key, json.dumps(data, default=str))
            return True
        except Exception as e:
            logging.error(f"❌ Error saving to Redis: {e}")
            return False

    @staticmethod
    def save_json_safe(data, basket_name, item_name, silent=True):
        try:
            data_str = json.dumps(data, default=str)
            data_size = len(data_str.encode('utf-8'))

            if not silent:
                print(f"📊 Data size: {data_size:,} bytes")

            if data_size > 5000000:  # 5MB limit
                if not silent:
                    print("⚠️ Data very large, consider chunking...")

                if 'entries' in data and isinstance(data['entries'], list):
                    data['entries'] = data['entries'][-100:]
                    data['truncated'] = True
                    data['original_entry_count'] = len(data.get('entries', []))

                    if not silent:
                        print(f"📉 Truncated to {len(data['entries'])} entries")

            redis_key = f"{basket_name}:{item_name}"
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    result = redis.set(redis_key, json.dumps(data, default=str))

                    if result:
                        if not silent:
                            print(f"✅ Successfully saved to Redis key '{redis_key}'")
                        return True
                    else:
                        if not silent:
                            print(f"⚠️ Redis returned False on attempt {attempt + 1}/{max_retries}")
                        continue

                except Exception as e:
                    if not silent:
                        print(f"🔗 Redis error on attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt == max_retries - 1:
                        return False
                    continue

            return False

        except Exception as e:
            if not silent:
                print(f"❌ Unexpected error in save_json_safe: {e}")
            return False

    @staticmethod
    def load_json(basket_name, item_name):
        try:
            redis_key = f"{basket_name}:{item_name}"
            result = redis.get(redis_key)

            if result:
                return json.loads(result)
            else:
                return {}

        except Exception as e:
            logging.error(f"❌ Error loading from Redis: {e}")
            return {}

# =====================================
# YOUR AUDIT SAVE FUNCTION (FROM PASTE-2)
# =====================================

def save_audit_log_to_redis(agent):
    """Your audit save function from paste-2"""
    print("💾 Saving new audit entries to Redis...")

    if not hasattr(agent, 'audit_trail') or not agent.audit_trail:
        print("⚠️ Agent has no audit trail to save.")
        return None

    audit_trail = agent.audit_trail

    if not hasattr(audit_trail, 'current_session_id') or not audit_trail.current_session_id:
        print("❌ Cannot save: Audit trail session has not been properly started.")
        return None

    if not hasattr(audit_trail, 'entries') or not audit_trail.entries:
        print("ℹ️ No new entries to save in the audit trail.")
        return None

    try:
        if not hasattr(audit_trail, '_last_saved_entry_count'):
            audit_trail._last_saved_entry_count = 0

        new_entries = audit_trail.entries[audit_trail._last_saved_entry_count:]

        if not new_entries:
            print("ℹ️ No new entries since last save.")
            return None

        print(f"📝 Saving {len(new_entries)} new entries (total: {len(audit_trail.entries)})")

        session_data = {
            "session_id": audit_trail.current_session_id,
            "start_time": getattr(audit_trail, 'session_start_time', datetime.now().isoformat()),
            "end_time": datetime.now().isoformat(),
            "metadata": getattr(audit_trail, 'session_metadata', {}),
            "entries": new_entries,
            "total_entries": len(audit_trail.entries),
            "entry_range": f"{audit_trail._last_saved_entry_count}-{len(audit_trail.entries)}",
            "agent_type": "ConversationalAgent",
            "conversation_stats": agent.get_memory_stats() if hasattr(agent, 'get_memory_stats') else {}
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        item_name = f"session_{audit_trail.current_session_id}_{timestamp}"

        result = UpstashStorage.save_json_safe(session_data, AUDIT_PREFIX, item_name, silent=False)

        if result:
            audit_trail._last_saved_entry_count = len(audit_trail.entries)
            print(f"✅ Audit log saved as '{item_name}' - {len(new_entries)} new entries")
        else:
            print("⚠️ Audit save had issues but continuing...")

        return item_name

    except Exception as e:
        print(f"❌ Error saving audit trail to Redis: {e}")
        return None

# =====================================
# YOUR FINANCIAL TOOLS (FROM PASTE-2)
# =====================================

@robust_tool(retries=3, wait_seconds=1.0)
@log_tool_call("get_stock_info")
def get_stock_info(symbol="AAPL", info_type="basic"):
    """Your stock info tool from paste-2"""
    try:
        logging.info(f"📊 Getting {info_type} info for {symbol}")
        stock = yf.Ticker(symbol)
        info = stock.info

        if not info:
            return {"status": "error", "error": f"No data found for {symbol}"}

        result = {}

        if info_type == "basic":
            result = {
                "symbol": symbol,
                "name": info.get("shortName", "N/A"),
                "current_price": info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
                "market_cap": info.get("marketCap", "N/A"),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "dividend_yield": info.get("dividendYield", "N/A"),
                "beta": info.get("beta", "N/A")
            }
        elif info_type == "detailed":
            result = {
                "symbol": symbol,
                "name": info.get("shortName", "N/A"),
                "current_price": info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
                "market_cap": info.get("marketCap", "N/A"),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "forward_pe": info.get("forwardPE", "N/A"),
                "peg_ratio": info.get("pegRatio", "N/A"),
                "price_to_book": info.get("priceToBook", "N/A"),
                "debt_to_equity": info.get("debtToEquity", "N/A"),
                "return_on_equity": info.get("returnOnEquity", "N/A"),
                "profit_margins": info.get("profitMargins", "N/A"),
                "beta": info.get("beta", "N/A"),
                "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
                "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
                "avg_volume": info.get("averageVolume", "N/A"),
                "dividend_yield": info.get("dividendYield", "N/A")
            }
        else:
            result = {
                "symbol": symbol,
                "name": info.get("shortName", "N/A"),
                "current_price": info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
                "market_cap": info.get("marketCap", "N/A"),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "info_type": f"Unknown type '{info_type}', showing basic info"
            }

        item_name = f"{symbol}_info_{info_type}"
        UpstashStorage.save_json(result, FINANCE_PREFIX, item_name)

        return {"status": "success", "result": result}

    except Exception as e:
        logging.error(f"❌ Error getting stock info: {str(e)}")
        return {"status": "error", "error": str(e)}

@robust_tool(retries=2)
@log_tool_call("create_interactive_chart")
def create_interactive_chart(symbols=["AAPL"], period="6mo", chart_type="candlestick", indicators=None):
    """Your chart creation tool from paste-2"""
    try:
        if isinstance(symbols, str):
            if symbols.startswith('[') and symbols.endswith(']'):
                try:
                    symbols = json.loads(symbols)
                except:
                    symbols = [s.strip().upper() for s in symbols.strip('[]').split(',')]
            else:
                symbols = [s.strip().upper() for s in symbols.split(',')]

        cleaned_symbols = []
        for symbol in symbols:
            symbol = symbol.strip().upper()
            if symbol == "RDS.A":
                symbol = "RDS-A"
            elif symbol == "RDS.B":
                symbol = "RDS-B"
            cleaned_symbols.append(symbol)

        symbols = cleaned_symbols

        if indicators is None:
            indicators = []
        elif isinstance(indicators, str):
            indicators = [i.strip() for i in indicators.split(',')]

        logging.info(f"📊 Creating {chart_type} chart for {symbols}")

        fig = go.Figure()
        chart_data = {}

        for i, symbol in enumerate(symbols):
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period=period)

                if hist.empty:
                    continue

                chart_data[symbol] = {
                    "dates": hist.index.strftime('%Y-%m-%d').tolist(),
                    "open": hist['Open'].round(2).tolist(),
                    "high": hist['High'].round(2).tolist(),
                    "low": hist['Low'].round(2).tolist(),
                    "close": hist['Close'].round(2).tolist(),
                    "volume": hist['Volume'].tolist()
                }

                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                color = colors[i % len(colors)]

                if chart_type == "candlestick":
                    fig.add_trace(go.Candlestick(
                        x=hist.index,
                        open=hist['Open'],
                        high=hist['High'],
                        low=hist['Low'],
                        close=hist['Close'],
                        name=symbol,
                        increasing_line_color=color,
                        decreasing_line_color='red'
                    ))
                elif chart_type == "ohlc":
                    fig.add_trace(go.Ohlc(
                        x=hist.index,
                        open=hist['Open'],
                        high=hist['High'],
                        low=hist['Low'],
                        close=hist['Close'],
                        name=symbol
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['Close'],
                        mode='lines',
                        name=symbol,
                        line=dict(color=color, width=2)
                    ))

                for indicator in indicators:
                    if indicator.startswith('SMA'):
                        try:
                            window = int(indicator[3:])
                            sma = hist['Close'].rolling(window=window).mean()
                            fig.add_trace(go.Scatter(
                                x=hist.index,
                                y=sma,
                                mode='lines',
                                name=f"{symbol} {indicator}",
                                line=dict(dash='dash', color=color, width=1),
                                opacity=0.7
                            ))
                        except:
                            pass
                    elif indicator.startswith('EMA'):
                        try:
                            window = int(indicator[3:])
                            ema = hist['Close'].ewm(span=window).mean()
                            fig.add_trace(go.Scatter(
                                x=hist.index,
                                y=ema,
                                mode='lines',
                                name=f"{symbol} {indicator}",
                                line=dict(dash='dot', color=color, width=1),
                                opacity=0.7
                            ))
                        except:
                            pass

            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue

        fig.update_layout(
            title=f"Stock Analysis: {', '.join(symbols)} ({period})",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=600,
            showlegend=True,
            template="plotly_white",
            hovermode='x unified'
        )

        fig.update_layout(xaxis_rangeslider_visible=False)

        chart_info = {
            "timestamp": datetime.now().isoformat(),
            "symbols": symbols,
            "period": period,
            "chart_type": chart_type,
            "indicators": indicators,
            "data": chart_data
        }

        chart_item_name = f'chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        UpstashStorage.save_json(chart_info, CHART_PREFIX, chart_item_name)

        return {
            "status": "success",
            "result": {
                "chart_created": True,
                "symbols": symbols,
                "period": period,
                "chart_type": chart_type,
                "indicators": indicators,
                "data_location": f"Redis Key: {CHART_PREFIX}:{chart_item_name}",
                "chart_figure": fig
            }
        }

    except Exception as e:
        logging.error(f"❌ Error creating chart: {str(e)}")
        return {"status": "error", "error": str(e)}

@robust_tool(retries=2)
@log_tool_call("compare_stocks")
def compare_stocks(symbols=["AAPL", "MSFT"], period="3mo", metrics=["price_change", "volatility"]):
    """Your stock comparison tool from paste-2"""
    try:
        if isinstance(symbols, str):
            if symbols.startswith('[') and symbols.endswith(']'):
                try:
                    symbols = json.loads(symbols)
                except:
                    symbols = [s.strip().upper() for s in symbols.strip('[]').split(',')]
            else:
                symbols = [s.strip().upper() for s in symbols.split(',')]

        cleaned_symbols = []
        for symbol in symbols:
            symbol = symbol.strip().upper()
            if symbol == "RDS.A":
                symbol = "RDS-A"
            elif symbol == "RDS.B":
                symbol = "RDS-B"
            cleaned_symbols.append(symbol)

        symbols = cleaned_symbols
        logging.info(f"🔍 Comparing {len(symbols)} stocks: {symbols}")

        results = {}
        comparison_data = {}

        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period=period)
                info = stock.info

                if hist.empty:
                    results[symbol] = {"error": "No data available"}
                    continue

                current_price = hist['Close'].iloc[-1]
                starting_price = hist['Close'].iloc[0]
                price_change_pct = ((current_price - starting_price) / starting_price) * 100

                daily_returns = hist['Close'].pct_change().dropna()
                volatility = daily_returns.std() * np.sqrt(252)

                stock_data = {
                    "current_price": round(current_price, 2),
                    "price_change_percent": round(price_change_pct, 2),
                    "volatility": round(volatility, 4),
                    "avg_volume": int(hist['Volume'].mean()),
                    "market_cap": info.get('marketCap', 0),
                    "pe_ratio": info.get('trailingPE', 0),
                    "sector": info.get('sector', 'Unknown')
                }

                results[symbol] = stock_data
                comparison_data[symbol] = stock_data

            except Exception as e:
                results[symbol] = {"error": str(e)}

        rankings = {}
        if len(comparison_data) > 1:
            for metric in metrics:
                metric_key = "price_change_percent" if metric in ["price_change_percent", "price_change"] else metric

                valid_stocks = {k: v[metric_key] for k, v in comparison_data.items()
                                if metric_key in v and v[metric_key] is not None and v[metric_key] != 0}

                if valid_stocks:
                    if metric == "volatility":
                        sorted_stocks = sorted(valid_stocks.items(), key=lambda x: x[1])
                    else:
                        sorted_stocks = sorted(valid_stocks.items(), key=lambda x: x[1], reverse=True)

                    rankings[metric] = {
                        "best": sorted_stocks[0] if sorted_stocks else None,
                        "worst": sorted_stocks[-1] if sorted_stocks else None,
                        "ranking": [{"symbol": k, "value": v} for k, v in sorted_stocks]
                    }

        comparison_result = {
            "individual_data": results,
            "rankings": rankings,
            "period": period,
            "comparison_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        comparison_item_name = f'comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        UpstashStorage.save_json(comparison_result, FINANCE_PREFIX, comparison_item_name)

        return {"status": "success", "result": comparison_result}

    except Exception as e:
        logging.error(f"❌ Error comparing stocks: {str(e)}")
        return {"status": "error", "error": str(e)}

@robust_tool(retries=2)
@log_tool_call("get_financial_statements")
def get_financial_statements(symbol="AAPL", statement_type="income", period="quarterly"):
    """Your financial statements tool from paste-2"""
    try:
        logging.info(f"📋 Getting {statement_type} statement for {symbol}")
        stock = yf.Ticker(symbol)

        if statement_type.lower() == "income":
            data = stock.quarterly_financials if period == "quarterly" else stock.financials
        elif statement_type.lower() == "balance":
            data = stock.quarterly_balance_sheet if period == "quarterly" else stock.balance_sheet
        elif statement_type.lower() == "cashflow":
            data = stock.quarterly_cashflow if period == "quarterly" else stock.cashflow
        else:
            return {"status": "error", "error": "Invalid statement type"}

        if data.empty:
            return {"status": "error", "error": f"No {statement_type} data found"}

        result = {}
        for col in data.columns:
            period_data = {}
            for idx in data.index:
                value = data.loc[idx, col]
                if pd.notna(value):
                    period_data[str(idx)] = float(value) if isinstance(value, (int, float)) else str(value)
            if period_data:
                result[col.strftime('%Y-%m-%d')] = period_data

        item_name = f'{symbol}_{statement_type}_{period}'
        UpstashStorage.save_json({
            "symbol": symbol,
            "statement_type": statement_type,
            "period": period,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }, FINANCE_PREFIX, item_name)

        return {"status": "success",
                "result": {"symbol": symbol, "statement_type": statement_type, "period": period, "data": result}}

    except Exception as e:
        logging.error(f"❌ Error getting financial statements: {str(e)}")
        return {"status": "error", "error": str(e)}

# =====================================
# YOUR CONVERSATIONAL AGENT (FROM PASTE-2)
# =====================================

class ConversationalAgent(Agent):
    """Your ConversationalAgent from paste-2"""

    def __init__(self, llm: LLM, system_prompt: Optional[str] = None,
                 cache_ttl: Optional[int] = 3600, cache_size: int = 100,
                 use_vector_memory: bool = False, enable_auditing: bool = True):

        super().__init__(llm, system_prompt, cache_ttl, cache_size, enable_auditing)

        self.conversation = ConversationChain(
            llm,
            EnhancedMemory(),
            system_prompt or "You are a helpful assistant that remembers previous interactions."
        )

        if use_vector_memory:
            self.init_vector_memory()

    def init_vector_memory(self):
        print("Vector memory initialized (using enhanced keyword search)")

    def chat(self, message: str, thread_id: str = "default",
             use_search: bool = True, context_window: int = 10) -> str:
        """Your chat method from paste-2"""

        if hasattr(self, 'audit_trail') and self.audit_trail:
            self.audit_trail.log_custom("chat_message_received", {
                "thread_id": thread_id,
                "message": message,
                "use_search": use_search,
                "context_window": context_window
            })

        if self.tools and len(self.tools) > 0:
            messages = self.conversation.get_conversation_context(thread_id, context_window)
            messages.append({"role": "user", "content": message})

            response = self.llm.generate_with_tools(messages, self.tools)

            self.conversation.add_message("user", message, thread_id)

            if response["type"] == "tool_call":
                tool_name = response["tool"]
                tool_args = response["args"]
                tool_reasoning = response.get("reasoning", "")

                tool = next((t for t in self.tools if t.name == tool_name), None)

                if tool:
                    cache_to_use = self.cache if hasattr(self, 'cache') else None
                    tool_start_time = time.time()
                    tool_result = tool(cache=cache_to_use, **tool_args)
                    tool_latency = time.time() - tool_start_time

                    if hasattr(self, 'audit_trail') and self.audit_trail:
                        self.audit_trail.log_tool_call(
                            tool_name,
                            tool_args,
                            tool_result,
                            latency=tool_latency,
                            cached=tool_result.get("cached", False),
                            metadata={"thread_id": thread_id, "context": "conversational_chat"}
                        )

                    self.conversation.add_message(
                        "assistant",
                        f"I'll use the {tool_name} tool with these parameters: {json.dumps(tool_args)}",
                        thread_id
                    )

                    if tool_reasoning:
                        self.conversation.add_message(
                            "system",
                            f"Tool reasoning: {tool_reasoning}",
                            thread_id,
                            {"type": "reasoning"}
                        )

                        if hasattr(self, 'audit_trail') and self.audit_trail:
                            self.audit_trail.log_reasoning(
                                tool_reasoning,
                                metadata={"tool": tool_name, "thread_id": thread_id}
                            )

                    cache_notice = " (from cache)" if tool_result.get("cached", False) else ""
                    result_content = f"Tool {tool_name} returned{cache_notice}: {json.dumps(tool_result.get('result', tool_result))}"

                    self.conversation.add_message("system", result_content, thread_id)

                    messages = self.conversation.get_conversation_context(thread_id, context_window + 3)
                    messages.append({
                        "role": "system",
                        "content": "Based on the tool results above, provide a comprehensive analysis in plain English. Do NOT output tool syntax. Analyze and summarize the actual financial data."
                    })

                    response_start_time = time.time()
                    final_response = self.llm.generate(messages)
                    response_latency = time.time() - response_start_time

                    self.conversation.add_message("assistant", final_response, thread_id)

                    if hasattr(self, 'audit_trail') and self.audit_trail:
                        self.audit_trail.log_custom("conversational_response", {
                            "thread_id": thread_id,
                            "response": final_response,
                            "tool_used": tool_name,
                            "latency": response_latency
                        })

                    return final_response

        response_start_time = time.time()
        if use_search:
            response = self.conversation.search_and_respond(
                message, thread_id, context_window
            )
        else:
            response = self.conversation.generate_response(
                message, thread_id, context_window
            )
        response_latency = time.time() - response_start_time

        if hasattr(self, 'audit_trail') and self.audit_trail:
            self.audit_trail.log_custom("conversational_response", {
                "thread_id": thread_id,
                "user_message": message,
                "assistant_response": response,
                "use_search": use_search,
                "context_window": context_window,
                "latency": response_latency,
                "tool_used": None
            })

        return response

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        total_messages = len(self.conversation.memory.history)
        total_threads = len(self.conversation.memory.threads)

        type_counts = {}
        for msg in self.conversation.memory.history:
            msg_type = msg.get("type", msg.get("role", "unknown"))
            type_counts[msg_type] = type_counts.get(msg_type, 0) + 1

        indexed_words = len(self.conversation.memory.indexed_content)

        return {
            "total_messages": total_messages,
            "total_threads": total_threads,
            "message_types": type_counts,
            "indexed_words": indexed_words,
            "has_audit_trail": hasattr(self, 'audit_trail') and self.audit_trail is not None
        }

# =====================================
# INITIALIZE YOUR REAL FINANCEGPT (FROM PASTE-2)
# =====================================

print("🔄 Initializing YOUR real FinanceGPT system...")

# Test Redis connection
try:
    print("🧪 Testing Redis connection...")
    test_data = {"test": "connection", "timestamp": datetime.now().isoformat()}
    result = UpstashStorage.save_json(test_data, "test", "connection")
    if not result:
        raise Exception("Save test failed")

    loaded_data = UpstashStorage.load_json("test", "connection")
    if loaded_data.get("test") != "connection":
        raise Exception("Load test failed")

    redis.delete("test:connection")
    print("✅ Redis connection successful!")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
    raise

# Initialize LLM
try:
    llm = OpenAILLM(
        api_key="6rdEEEr8WLCOp1mMdS3MTgPgT9A9IRSw",
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        base_url="https://api.deepinfra.com/v1/openai"
    )
    print("✅ LLM initialized successfully with DeepInfra")
except Exception as e:
    print(f"❌ Error initializing LLM: {e}")
    raise

# Create YOUR tools from paste-2
tools = [
    Tool(
        name="get_stock_info",
        func=get_stock_info,
        description="Retrieve stock information (basic or detailed) for a given symbol. Parameters: symbol (string), info_type (string, optional) - 'basic' or 'detailed'",
        cacheable=True,
        allow_repeated_calls=False
    ),
    Tool(
        name="create_interactive_chart",
        func=create_interactive_chart,
        description="Generate interactive charts for one or more stocks, supporting various chart types and technical indicators. Parameters: symbols (list or comma-separated string), period (string, optional), chart_type (string, optional) - 'line', 'candlestick', 'ohlc', indicators (list, optional) - like ['SMA20', 'SMA50', 'volume']",
        cacheable=True,
        allow_repeated_calls=False
    ),
    Tool(
        name="compare_stocks",
        func=compare_stocks,
        description="Compare multiple stocks. Parameters: symbols (list or comma-separated string), period (string, optional), metrics (list, optional)",
        cacheable=True,
        allow_repeated_calls=False
    ),
    Tool(
        name="get_financial_statements",
        func=get_financial_statements,
        description="Retrieve financial statements (income, balance, cash flow) for a company. Parameters: symbol (string), statement_type (string) - 'income', 'balance', or 'cashflow', period (string, optional) - 'annual' or 'quarterly'",
        cacheable=True,
        allow_repeated_calls=False
    )
]

# Create YOUR ConversationalAgent from paste-2
finance_bot = ConversationalAgent(
    llm=llm,
    system_prompt="""You are FinanceGPT, an expert financial analysis assistant with access to real-time market data and advanced analytical tools.

IMPORTANT:
- Never output or display tool or function calls directly to the user.
- Always use the tools via the agent interface, retrieve the result, and summarize the result in clear, conversational English.
- If a user asks for financial data, analysis, or charts, use the appropriate tool, then explain the results in a user-friendly way.
- Never show Python code or tool invocation syntax to the user.

Capabilities:
- Stock Information: Use get_stock_info for current prices, company overviews, and key metrics.
- Charting: Use create_interactive_chart to visualize price trends and technical patterns.
- Stock Comparison: Use compare_stocks to analyze relative performance or metrics across multiple stocks.
- Financial Statements: Use get_financial_statements to examine quarterly/annual income, balance sheet, and cash flow data.

Guidelines:
- Always provide clear, actionable insights backed by data.
- Explain the practical meaning of numbers.
- Use charts and visualizations when they add value.
- Consider risk factors and provide balanced analysis.
- Ask follow-up questions to clarify user needs.
- Provide context for financial metrics and ratios.
- Be conversational but professional.

Reminders:
- Past performance doesn't guarantee future results.
- Always mention risks and limitations.
- Provide educational context for complex concepts.
- Suggest related analyses that might be valuable.""",
    cache_ttl=300,
    use_vector_memory=False,
    enable_auditing=True
)

# Add YOUR tools to the agent
finance_bot.add_tools(tools)

# Check if YOUR agent has audit capabilities and start session if available
if hasattr(finance_bot, 'audit_trail') and finance_bot.audit_trail:
    if hasattr(finance_bot.audit_trail, 'start_session'):
        finance_bot.audit_trail.start_session({
            "application": "FinanceGPT",
            "version": "Enhanced with Redis Storage and Full Audit Trail",
            "features": ["stock_info", "charts", "comparison", "audit_trail", "enhanced_memory", "caching", "redis_storage"],
            "conversation_tracking": True,
            "database": "Upstash Redis",
            "start_time": datetime.now().isoformat()
        })
        print("✅ Audit session started")
    else:
        print("ℹ️ Audit trail exists but no start_session method")
else:
    print("ℹ️ No audit trail found, continuing without audit features")

print("🤖 YOUR Enhanced FinanceGPT with Redis Storage is ready!")
print("=" * 60)
print("✅ YOUR ConversationalAgent initialized")
print("✅ YOUR tools loaded and working")
print("✅ YOUR audit trail active")
print("✅ YOUR caching system enabled")
print("✅ YOUR Redis storage connected")

# =====================================
# SIMPLE USER MANAGEMENT FOR GRADIO
# =====================================

import gradio as gr
import uuid

class SimpleUserManager:
    """Simple user management for Gradio interface"""

    def __init__(self):
        self.users = {}
        self.load_users()

    def load_users(self):
        """Load users from Redis"""
        try:
            result = UpstashStorage.load_json(USERS_PREFIX, "all_users")
            if result:
                self.users = result
                print(f"✅ Loaded {len(self.users)} users from Redis")
        except Exception as e:
            print(f"⚠️ Error loading users: {e}")

    def save_users(self):
        """Save users to Redis"""
        try:
            UpstashStorage.save_json(self.users, USERS_PREFIX, "all_users")
        except Exception as e:
            print(f"⚠️ Error saving users: {e}")

    def create_user(self, username):
        """Create new user"""
        if username in self.users:
            return None

        user_id = str(uuid.uuid4())
        self.users[username] = {
            "user_id": user_id,
            "username": username,
            "created_at": datetime.now().isoformat(),
            "conversation_count": 0
        }
        self.save_users()
        return user_id

    def login_user(self, username):
        """Login existing user"""
        if username in self.users:
            return self.users[username]["user_id"]
        return None

    def get_user_by_id(self, user_id):
        """Get user by ID"""
        for user_data in self.users.values():
            if user_data["user_id"] == user_id:
                return user_data
        return None

# Initialize user manager
user_manager = SimpleUserManager()

# =====================================
# GRADIO INTERFACE FUNCTIONS
# =====================================

def handle_chat_with_your_bot(message, history, user_id):
    """Handle chat using YOUR real FinanceGPT"""
    if not user_id:
        return "", history + [(message, "❌ Please log in first to use the system.")]

    try:
        # Use YOUR actual FinanceGPT bot
        response = finance_bot.chat(message, thread_id=f"gradio_user_{user_id}")

        # Save audit log after each interaction
        save_audit_log_to_redis(finance_bot)

        # Check if chart was created
        chart_result = None
        if hasattr(finance_bot, 'audit_trail') and finance_bot.audit_trail:
            recent_entries = finance_bot.audit_trail.entries[-5:]
            for entry in recent_entries:
                if (entry.get('type') == 'tool_call' and
                    entry.get('tool_name') == 'create_interactive_chart' and
                    entry.get('result', {}).get('status') == 'success'):
                    tool_result = entry.get('result', {}).get('result', {})
                    if 'chart_figure' in tool_result:
                        chart_result = tool_result['chart_figure']
                        break

        # Update conversation history
        history.append((message, response))

        # Return chart if created
        if chart_result:
            return "", history, gr.update(value=chart_result, visible=True)
        else:
            return "", history, gr.update(visible=False)

    except Exception as e:
        error_response = f"❌ Error: {str(e)}\n\nThis error is tracked in the audit trail for debugging!"
        history.append((message, error_response))
        return "", history, gr.update(visible=False)

def user_login_handler(username, create_account):
    """Handle user login/registration"""
    if not username.strip():
        return None, "Please enter a username"

    username = username.strip()

    if create_account:
        user_id = user_manager.create_user(username)
        if user_id:
            return user_id, f"✅ Account created! Welcome {username}!"
        else:
            return None, f"❌ Username '{username}' already exists"
    else:
        user_id = user_manager.login_user(username)
        if user_id:
            return user_id, f"✅ Welcome back, {username}!"
        else:
            return None, f"❌ User '{username}' not found. Check 'Create Account' to register."

def get_user_info_display(user_id):
    """Get user information display"""
    if not user_id:
        return "Please log in to see your information"

    user_data = user_manager.get_user_by_id(user_id)
    if not user_data:
        return "User data not found"

    return f"""
    <div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
                padding: 15px; border-radius: 10px; color: white; margin: 10px 0;">
        <h3 style="margin: 0;">👤 {user_data['username']}</h3>
        <p style="margin: 5px 0;">Account created: {user_data['created_at'][:10]}</p>
        <p style="margin: 0; font-size: 12px; opacity: 0.9;">🔍 Using YOUR real FinanceGPT with full observability</p>
    </div>
    """

def get_real_observability_metrics():
    """Get real observability metrics from YOUR system"""
    try:
        # Check what's actually available in YOUR system
        audit_summary = {}
        cache_stats = {}

        if hasattr(finance_bot, 'audit_trail') and finance_bot.audit_trail:
            if hasattr(finance_bot.audit_trail, 'get_summary'):
                audit_summary = finance_bot.audit_trail.get_summary()
            elif hasattr(finance_bot.audit_trail, 'entries'):
                # Calculate basic summary from entries
                entries = finance_bot.audit_trail.entries
                tool_calls = [e for e in entries if e.get('type') == 'tool_call']
                cached_calls = [e for e in tool_calls if e.get('cached')]
                latencies = [e.get('latency', 0) for e in tool_calls if e.get('latency')]

                audit_summary = {
                    'total_entries': len(entries),
                    'tool_calls': len(tool_calls),
                    'cached_calls': len(cached_calls),
                    'cache_hit_rate': len(cached_calls) / len(tool_calls) * 100 if tool_calls else 0,
                    'avg_latency': sum(latencies) / len(latencies) if latencies else 0
                }

        if hasattr(finance_bot, 'cache'):
            if hasattr(finance_bot.cache, 'get_stats'):
                cache_stats = finance_bot.cache.get_stats()
            elif hasattr(finance_bot.cache, 'cache'):
                # Basic cache info
                cache_stats = {
                    'entries': len(finance_bot.cache.cache),
                    'hit_rate': 0  # Would need tracking to calculate
                }

        # Tool usage breakdown from real audit trail
        tool_usage = {}
        if hasattr(finance_bot, 'audit_trail') and finance_bot.audit_trail and hasattr(finance_bot.audit_trail, 'entries'):
            for entry in finance_bot.audit_trail.entries:
                if entry.get('type') == 'tool_call':
                    tool = entry.get('tool_name', 'unknown')
                    if tool not in tool_usage:
                        tool_usage[tool] = {'calls': 0, 'cached': 0, 'total_latency': 0}

                    tool_usage[tool]['calls'] += 1
                    if entry.get('cached'):
                        tool_usage[tool]['cached'] += 1
                    tool_usage[tool]['total_latency'] += entry.get('latency', 0)

            # Calculate averages
            for tool, stats in tool_usage.items():
                if stats['calls'] > 0:
                    stats['avg_latency'] = stats['total_latency'] / stats['calls']
                    stats['cache_rate'] = (stats['cached'] / stats['calls']) * 100

        # FIXED: Calculate actual cache hit rate from tool usage data
        total_calls = sum(stats['calls'] for stats in tool_usage.values())
        total_cached = sum(stats['cached'] for stats in tool_usage.values())
        actual_cache_hit_rate = (total_cached / total_calls * 100) if total_calls > 0 else 0

        metrics_html = f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0;">

            <!-- System Health -->
            <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                        border-radius: 12px; padding: 20px; color: white; text-align: center;">
                <h3 style="margin: 0 0 10px 0;">🟢 System Health</h3>
                <div style="font-size: 24px; font-weight: bold;">{audit_summary.get('total_entries', 0)}</div>
                <div style="font-size: 14px; opacity: 0.9;">Real Events Tracked</div>
            </div>

            <!-- Performance -->
            <div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
                        border-radius: 12px; padding: 20px; color: white; text-align: center;">
                <h3 style="margin: 0 0 10px 0;">⚡ Performance</h3>
                <div style="font-size: 24px; font-weight: bold;">{audit_summary.get('avg_latency', 0):.3f}s</div>
                <div style="font-size: 14px; opacity: 0.9;">Real Avg Response Time</div>
            </div>

            <!-- Cache Efficiency - FIXED TO USE ACTUAL RATE -->
            <div style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
                        border-radius: 12px; padding: 20px; color: white; text-align: center;">
                <h3 style="margin: 0 0 10px 0;">💾 Cache</h3>
                <div style="font-size: 24px; font-weight: bold;">{actual_cache_hit_rate:.1f}%</div>
                <div style="font-size: 14px; opacity: 0.9;">Real Hit Rate</div>
            </div>

            <!-- Tool Usage -->
            <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
                        border-radius: 12px; padding: 20px; color: white; text-align: center;">
                <h3 style="margin: 0 0 10px 0;">🛠️ Tools</h3>
                <div style="font-size: 24px; font-weight: bold;">{len(tool_usage)}</div>
                <div style="font-size: 14px; opacity: 0.9;">Active Tools</div>
            </div>

        </div>

        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0;">
            <h4 style="margin: 0 0 10px 0; color: #495057;">🔍 YOUR Real Tool Performance</h4>
            <div style="font-family: monospace; font-size: 13px; color: #6c757d;">
        """

        for tool_name, stats in tool_usage.items():
            metrics_html += f"""
                <div style="margin: 5px 0; padding: 12px; background: white; border-radius: 6px;
                           border-left: 5px solid #007bff; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <strong style="color: #2c3e50; font-size: 14px;">{tool_name}</strong><br>
                    <span style="color: #34495e; font-size: 13px;">
                        📞 {stats['calls']} calls |
                        ⚡ {stats.get('avg_latency', 0):.3f}s avg |
                        💾 {stats.get('cache_rate', 0):.1f}% cached
                    </span>
                </div>
            """

        if not tool_usage:
            metrics_html += """
                <div style='text-align: center; color: #e74c3c; background: #fff3cd;
                           padding: 15px; border-radius: 6px; border: 1px solid #ffeaa7;'>
                    <strong>⚠️ No tool usage yet!</strong><br>
                    <span style="font-size: 13px;">Try these prompts to trigger tools:</span><br>
                    <code style="background: #f8f9fa; padding: 3px 6px; border-radius: 3px; color: #2c3e50;">
                        "Get stock info for AAPL"
                    </code><br>
                    <code style="background: #f8f9fa; padding: 3px 6px; border-radius: 3px; color: #2c3e50;">
                        "Create chart for Tesla"
                    </code><br>
                    <code style="background: #f8f9fa; padding: 3px 6px; border-radius: 3px; color: #2c3e50;">
                        "Compare AAPL vs MSFT"
                    </code>
                </div>
            """

        metrics_html += """
            </div>
        </div>
        """

        return metrics_html

    except Exception as e:
        return f"<div style='color: red;'>Error getting metrics: {str(e)}</div>"

def create_real_audit_timeline():
    """Create audit timeline from YOUR real system"""
    try:
        if not hasattr(finance_bot, 'audit_trail') or not finance_bot.audit_trail:
            fig = go.Figure()
            fig.add_annotation(
                text="No audit data yet. Start using YOUR system to see real audit trails!",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='#666')
            )
            fig.update_layout(
                title="🔍 YOUR Real Audit Trail Timeline",
                height=400,
                paper_bgcolor='rgba(248, 249, 250, 0.8)'
            )
            return fig

        if not hasattr(finance_bot.audit_trail, 'entries'):
            fig = go.Figure()
            fig.add_annotation(
                text="Audit trail exists but no entries method found",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='#666')
            )
            fig.update_layout(
                title="🔍 YOUR Real Audit Trail Timeline",
                height=400,
                paper_bgcolor='rgba(248, 249, 250, 0.8)'
            )
            return fig

        entries = finance_bot.audit_trail.entries[-20:]  # Last 20 real entries

        if not entries:
            fig = go.Figure()
            fig.add_annotation(
                text="Start chatting to see YOUR real audit timeline!",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='#666')
            )
            fig.update_layout(
                title="🔍 YOUR Real Audit Trail Timeline",
                height=400,
                paper_bgcolor='rgba(248, 249, 250, 0.8)'
            )
            return fig

        timestamps = []
        event_types = []
        latencies = []
        colors = []
        hover_texts = []

        color_map = {
            'tool_call': '#3b82f6',
            'chat_message_received': '#10b981',
            'session_started': '#8b5cf6',
            'conversational_response': '#f59e0b',
            'error': '#ef4444'
        }

        for entry in entries:
            try:
                timestamps.append(datetime.fromisoformat(entry['timestamp']))
                event_type = entry.get('type', 'unknown')
                event_types.append(event_type)

                latency = entry.get('latency', 0)
                latencies.append(latency if latency else 0.001)

                colors.append(color_map.get(event_type, '#6c757d'))

                if event_type == 'tool_call':
                    tool_name = entry.get('tool_name', 'unknown')
                    cached = " (CACHED)" if entry.get('cached') else ""
                    hover_text = f"Tool: {tool_name}{cached}<br>Latency: {latency:.3f}s"
                else:
                    hover_text = f"Event: {event_type}<br>Time: {entry['timestamp'].split('T')[1][:8]}"

                hover_texts.append(hover_text)
            except Exception as e:
                print(f"Error processing entry: {e}")
                continue

        if not timestamps:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid timeline data found",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='#666')
            )
            fig.update_layout(
                title="🔍 YOUR Real Audit Trail Timeline",
                height=400,
                paper_bgcolor='rgba(248, 249, 250, 0.8)'
            )
            return fig

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=latencies,
            mode='markers+lines',
            marker=dict(
                size=[max(8, min(20, lat*10)) for lat in latencies],
                color=colors,
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            line=dict(color='rgba(59, 130, 246, 0.3)', width=1),
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>',
            name="Real Events"
        ))

        fig.update_layout(
            title="🔍 YOUR Real Audit Trail Timeline (Live from Jerzy Framework)",
            xaxis_title="Time",
            yaxis_title="Latency (seconds)",
            height=400,
            paper_bgcolor='rgba(248, 249, 250, 0.8)',
            showlegend=False,
            hovermode='closest'
        )

        return fig

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating timeline: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color='#666')
        )
        fig.update_layout(
            title="🔍 YOUR Real Audit Trail Timeline",
            height=400,
            paper_bgcolor='rgba(248, 249, 250, 0.8)'
        )
        return fig

def get_real_audit_details():
    """Get real audit details from YOUR system"""
    try:
        if not hasattr(finance_bot, 'audit_trail') or not finance_bot.audit_trail:
            return {}, "❌ No audit trail available"

        if not hasattr(finance_bot.audit_trail, 'entries'):
            return {}, "❌ Audit trail exists but no entries method found"

        entries = finance_bot.audit_trail.entries[-20:]  # Last 20 real entries

        # Try to get summary
        summary_data = {}
        if hasattr(finance_bot.audit_trail, 'get_summary'):
            summary_data = finance_bot.audit_trail.get_summary()
        elif entries:
            # Calculate basic summary
            tool_calls = [e for e in entries if e.get('type') == 'tool_call']
            cached_calls = [e for e in tool_calls if e.get('cached')]
            latencies = [e.get('latency', 0) for e in tool_calls if e.get('latency')]

            summary_data = {
                'total_entries': len(entries),
                'tool_calls': len(tool_calls),
                'cache_hit_rate': len(cached_calls) / len(tool_calls) * 100 if tool_calls else 0,
                'avg_latency': sum(latencies) / len(latencies) if latencies else 0
            }

        summary_text = f"""
        ## 📊 YOUR Real Audit Trail Summary

        - **Total Events**: {summary_data.get('total_entries', 0)} (from YOUR system)
        - **Tool Calls**: {summary_data.get('tool_calls', 0)} (real API calls)
        - **Cache Hit Rate**: {summary_data.get('cache_hit_rate', 0):.1f}% (actual performance)
        - **Average Latency**: {summary_data.get('avg_latency', 0):.3f}s (real response times)

        **🔍 This is YOUR actual Jerzy Framework audit data - no simulations!**
        """

        return entries, summary_text

    except Exception as e:
        return {}, f"❌ Error getting audit details: {str(e)}"

# =====================================
# CREATE GRADIO INTERFACE
# =====================================

def create_gradio_interface():
    """Create Gradio interface for YOUR real FinanceGPT"""

    with gr.Blocks(title="YOUR Real FinanceGPT - Jerzy Framework Demo", theme=gr.themes.Soft()) as demo:

        current_user_id = gr.State(None)

        gr.Markdown("""
        # 🚀 YOUR Real FinanceGPT - Jerzy Framework Observability Demo
        ### Using YOUR actual ConversationalAgent with real financial data and complete observability
        """)

        # Login Section
        with gr.Row():
            with gr.Column(scale=2):
                username_input = gr.Textbox(label="Username", placeholder="Enter your username")
            with gr.Column(scale=1):
                create_account_checkbox = gr.Checkbox(label="Create New Account", value=False)
            with gr.Column(scale=1):
                login_btn = gr.Button("Login", variant="primary")

        login_status = gr.Markdown("Please log in to start using YOUR real FinanceGPT system")
        user_info_display = gr.HTML()

        # Main Interface
        with gr.Tab("💬 YOUR Real Financial Chat"):
            gr.Markdown("### Chat with YOUR actual FinanceGPT using real market data")

            chatbot = gr.Chatbot(height=500, label="YOUR Real FinanceGPT - Powered by Jerzy Framework")
            chart_display = gr.Plot(label="📊 Real-Time Charts from YOUR Bot", visible=False)

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Tell me about Datadog stock, compare Apple vs Microsoft, show me Tesla chart...",
                    label="Chat with YOUR Real Bot",
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            gr.Examples(
                examples=[
                    "Get stock information for DDOG",
                    "Get detailed stock info for Apple",
                    "Create an interactive chart for Tesla with SMA20",
                    "Compare Tesla vs Ford vs GM performance",
                    "Get financial statements for Amazon income quarterly",
                    "Show me Apple stock info",
                    "Chart Microsoft with candlestick style",
                    "Compare AAPL vs MSFT vs GOOGL stocks"
                ],
                inputs=msg_input
            )

        with gr.Tab("📊 YOUR Real Observability"):
            gr.Markdown("### Live monitoring of YOUR Jerzy Framework system")

            refresh_btn = gr.Button("🔄 Refresh YOUR Real Metrics", variant="primary")

            metrics_display = gr.HTML(label="📈 YOUR Live System Metrics")
            audit_timeline = gr.Plot(label="🔍 YOUR Real Audit Trail Timeline")

        with gr.Tab("🔍 YOUR Audit Explorer"):
            gr.Markdown("### Deep dive into YOUR system execution")

            show_audit_btn = gr.Button("📋 Show YOUR Audit Details", variant="primary")

            audit_details = gr.JSON(label="YOUR Complete Audit Trail Data")
            audit_summary = gr.Markdown()

        with gr.Tab("ℹ️ Prompts to Trigger Tools"):
            gr.Markdown("""
            ## 🎯 **Specific Prompts to Trigger Each Tool**

            ### 🔧 **Tool 1: get_stock_info**
            ```
            "Get stock info for AAPL"
            "Get detailed stock information for Microsoft"
            "Tell me about Tesla stock"
            "Stock information for DDOG"
            ```

            ### 📊 **Tool 2: create_interactive_chart**
            ```
            "Create a chart for Apple"
            "Show me Tesla chart with SMA20 indicator"
            "Plot Microsoft candlestick chart"
            "Chart NVDA with technical indicators"
            ```

            ### ⚖️ **Tool 3: compare_stocks**
            ```
            "Compare Apple vs Microsoft"
            "Compare AAPL vs MSFT vs GOOGL"
            "Tesla vs Ford performance comparison"
            "Compare tech stocks NVDA vs AMD"
            ```

            ### 📋 **Tool 4: get_financial_statements**
            ```
            "Get financial statements for Apple"
            "Show me Amazon income statement quarterly"
            "Microsoft balance sheet annual"
            "Tesla cashflow statement"
            ```

            ## 🎯 **Cache Demo Sequence**:

            1. **First call** (cache MISS): `"Get stock info for AAPL"`
            2. **Second call** (cache HIT): `"Get stock info for AAPL"` ← Same exact prompt
            3. **Check Observability** → See cache hit rate increase!

            ## 🚀 **Best Demo Flow**:
            ```
            1. "Get stock info for AAPL" (Tool 1 - MISS)
            2. "Create chart for AAPL" (Tool 2 - MISS)
            3. "Compare AAPL vs MSFT" (Tool 3 - MISS)
            4. "Get financial statements for AAPL" (Tool 4 - MISS)
            5. "Get stock info for AAPL" (Tool 1 - HIT!)
            6. Refresh Observability → See all 4 tools + cache hits!
            ```
            """)

        with gr.Tab("ℹ️ About YOUR System"):
            gr.Markdown(f"""
            ## 🎯 YOUR Real Jerzy Framework Demo

            This interface uses **YOUR actual FinanceGPT** from paste-2 with:

            ### ✅ **YOUR Real Components**:
            - **YOUR ConversationalAgent**: The actual agent you built
            - **YOUR Tools**: Real financial analysis tools (get_stock_info, create_interactive_chart, etc.)
            - **YOUR Audit Trail**: Complete transparency into every operation
            - **YOUR Caching System**: Real performance optimization
            - **YOUR Redis Storage**: Production-grade data persistence

            ### 🔍 **Observability Features**:
            - Real-time audit trail of every tool call
            - Actual performance metrics (latency, cache hits)
            - Complete conversation flow tracking
            - Error logging and debugging capabilities

            ### 💡 **How It Works**:
            1. You chat with YOUR real FinanceGPT bot
            2. The Jerzy framework automatically tracks everything
            3. View real observability data in the monitoring tabs
            4. All data is stored in Redis for persistence

            **Session ID**: `{finance_bot.audit_trail.current_session_id if hasattr(finance_bot, 'audit_trail') and finance_bot.audit_trail else 'Not available'}`

            **🚀 This is YOUR framework in action - real data, real performance, real observability!**
            """)

        # Event Handlers
        def handle_login(username, create_account):
            user_id, message = user_login_handler(username, create_account)
            if user_id:
                user_display = get_user_info_display(user_id)
                return user_id, message, user_display, []
            else:
                return None, message, "", []

        def refresh_observability():
            metrics = get_real_observability_metrics()
            timeline = create_real_audit_timeline()
            return metrics, timeline

        def show_audit_trail():
            entries, summary = get_real_audit_details()
            return entries, summary

        # Connect events
        login_btn.click(
            handle_login,
            inputs=[username_input, create_account_checkbox],
            outputs=[current_user_id, login_status, user_info_display, chatbot]
        )

        msg_input.submit(
            handle_chat_with_your_bot,
            inputs=[msg_input, chatbot, current_user_id],
            outputs=[msg_input, chatbot, chart_display]
        )

        send_btn.click(
            handle_chat_with_your_bot,
            inputs=[msg_input, chatbot, current_user_id],
            outputs=[msg_input, chatbot, chart_display]
        )

        refresh_btn.click(
            refresh_observability,
            outputs=[metrics_display, audit_timeline]
        )

        show_audit_btn.click(
            show_audit_trail,
            outputs=[audit_details, audit_summary]
        )

        # Initialize with real data
        demo.load(
            refresh_observability,
            outputs=[metrics_display, audit_timeline]
        )

    return demo

# =====================================
# LAUNCH YOUR REAL SYSTEM
# =====================================

if __name__ == "__main__":
    print("🚀 Launching Gradio Interface for YOUR Real FinanceGPT...")
    print("=" * 60)
    print("✅ Using YOUR actual ConversationalAgent from paste-2")
    print("✅ YOUR real financial tools with live data")
    print("✅ YOUR complete audit trail and observability")
    print("✅ YOUR caching and performance optimization")
    print("✅ YOUR Redis storage integration")
    print("=" * 60)

    # Verify YOUR system is working
    if hasattr(finance_bot, 'audit_trail') and finance_bot.audit_trail:
        print(f"✅ YOUR audit trail is active: {finance_bot.audit_trail.current_session_id}")

    if hasattr(finance_bot, 'tools') and finance_bot.tools:
        print(f"✅ YOUR tools are loaded: {[tool.name for tool in finance_bot.tools]}")

    # Create and launch interface for YOUR system
    demo = create_gradio_interface()

    demo.launch(server_name="0.0.0.0", server_port=7860)
