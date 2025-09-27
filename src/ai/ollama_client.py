import requests
import json
import logging
from typing import Dict, Any, Optional, List
import time

from config.settings import settings

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self):
        self.host = settings.OLLAMA_HOST
        self.model = settings.OLLAMA_MODEL
        self.timeout = 120
        self.use_streaming = True  # Default to streaming for better timeout handling
        self.max_retries = 2
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            url = f"{self.host}/{endpoint}"
            response = requests.post(
                url,
                json=data,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"OLLAMA request failed: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"OLLAMA connection error: {e}")
            return None
    
    def check_connection(self) -> bool:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        try:
            response = self._make_request("api/tags", {})
            if response and 'models' in response:
                return [model['name'] for model in response['models']]
            return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def generate_response(self, prompt: str, context: Optional[str] = None, use_stream: bool = None) -> Optional[str]:
        """Generate response with streaming support and retry logic for better timeout handling"""
        if use_stream is None:
            use_stream = self.use_streaming
            
        # Try with retries
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"OLLAMA request attempt {attempt + 1}/{self.max_retries + 1}, timeout: {self.timeout}s, streaming: {use_stream}")
                
                if use_stream:
                    result = self._generate_response_stream(prompt, context)
                else:
                    result = self._generate_response_blocking(prompt, context)
                
                if result:
                    logger.info(f"OLLAMA request successful on attempt {attempt + 1}")
                    return result
                else:
                    logger.warning(f"OLLAMA request returned empty response on attempt {attempt + 1}")
                    
            except Exception as e:
                logger.error(f"OLLAMA request failed on attempt {attempt + 1}: {e}")
                
                # On last attempt, try the other method if streaming failed
                if attempt == self.max_retries and use_stream:
                    logger.info("Trying blocking method as final fallback")
                    try:
                        return self._generate_response_blocking(prompt, context)
                    except Exception as e2:
                        logger.error(f"Final blocking attempt also failed: {e2}")
        
        logger.error(f"All OLLAMA attempts failed after {self.max_retries + 1} tries")
        return None
    
    def _generate_response_blocking(self, prompt: str, context: Optional[str] = None) -> Optional[str]:
        """Original blocking response method"""
        try:
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            if context:
                data["context"] = context
            
            response = self._make_request("api/generate", data)
            
            if response and 'response' in response:
                return response['response']
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating blocking response: {e}")
            return None
    
    def _generate_response_stream(self, prompt: str, context: Optional[str] = None) -> Optional[str]:
        """Streaming response method for better timeout handling"""
        try:
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": True
            }
            
            if context:
                data["context"] = context
            
            url = f"{self.host}/api/generate"
            
            # Use streaming request with longer timeout
            response = requests.post(
                url,
                json=data,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'},
                stream=True
            )
            
            if response.status_code != 200:
                logger.error(f"OLLAMA streaming request failed: {response.status_code}")
                return None
            
            # Collect streaming response
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if 'response' in chunk:
                            full_response += chunk['response']
                        if chunk.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
            
            return full_response if full_response else None
            
        except requests.exceptions.Timeout:
            logger.error(f"OLLAMA streaming request timed out after {self.timeout} seconds")
            return None
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            return None

class AIAnalyzer:
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.scoring_weights = settings.SCORING_WEIGHTS
        self.score_ranges = settings.SCORE_RANGES
    
    def create_analysis_prompt(self, symbol: str, analysis_data: Dict[str, Any], news_data: str = None) -> str:
        prompt = f"""
        You are an expert financial analyst with deep expertise in technical analysis and market dynamics. 
        Analyze the following technical data for {symbol} and provide a comprehensive yet concise assessment.

        TECHNICAL DATA SUMMARY:
        - Current Price: ${analysis_data.get('current_price', 'N/A')}
        - Daily Trend: {analysis_data.get('daily_trend', 'N/A')}
        - Weekly Trend: {analysis_data.get('weekly_trend', 'N/A')}
        - RSI (14-period): {analysis_data.get('rsi', 'N/A')}
        - Volume Ratio: {analysis_data.get('volume_ratio', 'N/A')}x average
        - Price vs EMA20: {analysis_data.get('price_vs_ema20', 'N/A')}
        - Price vs EMA50: {analysis_data.get('price_vs_ema50', 'N/A')}
        - MACD Signal: {analysis_data.get('macd_signal', 'N/A')}
        - Volatility (Annualized): {analysis_data.get('volatility', 'N/A')}

        DETECTED PATTERNS:
        {json.dumps(analysis_data.get('patterns', {}), indent=2, default=str)}
        """
        
        # Add news data if available
        if news_data:
            prompt += f"""
        
        NEWS ANALYSIS:
        {news_data}
        """
        
        prompt += """
        
        ANALYSIS REQUIREMENTS:
        Provide a structured analysis in exactly 4-5 sentences covering:
        1. Overall trend assessment and strength (considering daily vs weekly alignment)
        2. Key technical indicators status (RSI levels, volume confirmation, moving average positioning)
        3. News sentiment and potential impact on price action (if news data provided)
        4. Risk-reward profile and notable patterns or signals
        5. Actionable short-term outlook with specific price levels if relevant

        Response should be professional, data-driven, and directly actionable for trading decisions.
        Consider both technical indicators and news sentiment in your analysis.
        Avoid speculation and focus on what the data clearly indicates.
        """
        return prompt
    
    def calculate_ai_score(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            scores = {}
            
            # Trend Alignment Score (30%)
            trend_score = self._calculate_trend_alignment_score(analysis_data)
            scores['trend_alignment'] = trend_score
            
            # Volume Confirmation Score (20%)
            volume_score = self._calculate_volume_score(analysis_data)
            scores['volume_confirmation'] = volume_score
            
            # RSI Positioning Score (15%)
            rsi_score = self._calculate_rsi_score(analysis_data)
            scores['rsi_positioning'] = rsi_score
            
            # Price Action Score (15%)
            price_score = self._calculate_price_action_score(analysis_data)
            scores['price_action'] = price_score
            
            # Volatility Analysis Score (10%)
            volatility_score = self._calculate_volatility_score(analysis_data)
            scores['volatility_analysis'] = volatility_score
            
            # Momentum Quality Score (10%)
            momentum_score = self._calculate_momentum_score(analysis_data)
            scores['momentum_quality'] = momentum_score
            
            # Calculate weighted total score (components are already 0-100, so no need to multiply by 100)
            total_score = (
                scores['trend_alignment'] * self.scoring_weights['trend_alignment'] +
                scores['volume_confirmation'] * self.scoring_weights['volume_confirmation'] +
                scores['rsi_positioning'] * self.scoring_weights['rsi_positioning'] +
                scores['price_action'] * self.scoring_weights['price_action'] +
                scores['volatility_analysis'] * self.scoring_weights['volatility_analysis'] +
                scores['momentum_quality'] * self.scoring_weights['momentum_quality']
            )
            
            # Determine signal classification
            signal_classification = self._get_signal_classification(total_score)
            
            return {
                'ai_score': round(min(total_score, 100), 2),  # 4-digit precision (XX.XX)
                'signal_classification': signal_classification,
                'component_scores': scores,
                'confidence': self._calculate_confidence(scores)
            }
            
        except Exception as e:
            logger.error(f"Error calculating AI score: {e}")
            return {
                'ai_score': 0,
                'signal_classification': 'Unknown',
                'component_scores': {},
                'confidence': 0
            }
    
    def _calculate_trend_alignment_score(self, data: Dict[str, Any]) -> float:
        try:
            daily_trend = data.get('daily_trend', {}).get('overall_trend', 'neutral')
            weekly_trend = data.get('weekly_trend', {}).get('overall_trend', 'neutral')
            alignment_score = data.get('alignment_score', 0)
            
            if daily_trend == weekly_trend and daily_trend != 'neutral':
                return min(100, 60 + (alignment_score * 40))
            elif daily_trend == 'neutral' or weekly_trend == 'neutral':
                return 40
            else:
                return 20
                
        except Exception:
            return 0
    
    def _calculate_volume_score(self, data: Dict[str, Any]) -> float:
        try:
            volume_ratio = data.get('volume_ratio', 1)
            
            if volume_ratio >= 2.0:
                return 100
            elif volume_ratio >= 1.5:
                return 80
            elif volume_ratio >= 1.2:
                return 60
            elif volume_ratio >= 1.0:
                return 40
            else:
                return 20
                
        except Exception:
            return 0
    
    def _calculate_rsi_score(self, data: Dict[str, Any]) -> float:
        try:
            rsi = data.get('rsi', 50)
            trend = data.get('daily_trend', {}).get('overall_trend', 'neutral')
            
            if trend == 'bullish':
                if 40 <= rsi <= 60:
                    return 100
                elif 30 <= rsi <= 70:
                    return 80
                elif rsi > 70:
                    return 40  # Overbought in uptrend
                else:
                    return 60
            elif trend == 'bearish':
                if 40 <= rsi <= 60:
                    return 100
                elif 30 <= rsi <= 70:
                    return 80
                elif rsi < 30:
                    return 40  # Oversold in downtrend
                else:
                    return 60
            else:
                return 50  # Neutral
                
        except Exception:
            return 0
    
    def _calculate_price_action_score(self, data: Dict[str, Any]) -> float:
        try:
            price_vs_ema20 = data.get('price_positioning', {}).get('price_vs_ema_20', 'below')
            price_vs_ema50 = data.get('price_positioning', {}).get('price_vs_ema_50', 'below')
            
            score = 0
            if price_vs_ema20 == 'above':
                score += 50
            if price_vs_ema50 == 'above':
                score += 50
            
            return score
            
        except Exception:
            return 0
    
    def _calculate_volatility_score(self, data: Dict[str, Any]) -> float:
        try:
            volatility = data.get('volatility', 0.5)
            
            # Optimal volatility range for trading
            if 0.15 <= volatility <= 0.4:
                return 100
            elif 0.1 <= volatility <= 0.6:
                return 80
            elif volatility <= 0.8:
                return 60
            else:
                return 30  # Too volatile
                
        except Exception:
            return 0
    
    def _calculate_momentum_score(self, data: Dict[str, Any]) -> float:
        try:
            macd_signal = data.get('macd_signal', 'neutral')
            patterns = data.get('patterns', {})
            
            score = 50  # Base score
            
            if macd_signal == 'bullish':
                score += 30
            elif macd_signal == 'bearish':
                score -= 30
            
            # Pattern bonuses
            if patterns.get('bullish_engulfing', False):
                score += 20
            if patterns.get('bearish_engulfing', False):
                score -= 20
            if patterns.get('breakout', False):
                score += 15
            if patterns.get('breakdown', False):
                score -= 15
            
            return max(0, min(100, score))
            
        except Exception:
            return 0
    
    def _get_signal_classification(self, score: float) -> str:
        for classification, (min_score, max_score) in self.score_ranges.items():
            if min_score <= score <= max_score:
                return classification.replace('_', ' ').title()
        return 'Unknown'
    
    def _calculate_confidence(self, scores: Dict[str, float]) -> float:
        try:
            # Confidence based on consistency of component scores
            score_values = list(scores.values())
            if not score_values:
                return 0
            
            mean_score = sum(score_values) / len(score_values)
            variance = sum((score - mean_score) ** 2 for score in score_values) / len(score_values)
            
            # Lower variance = higher confidence
            confidence = max(0, min(100, 100 - (variance / 10)))
            return round(confidence, 2)
            
        except Exception:
            return 0
    
    def analyze_stock(self, symbol: str, analysis_data: Dict[str, Any], news_data: str = None) -> Dict[str, Any]:
        try:
            # Calculate AI score
            ai_results = self.calculate_ai_score(analysis_data)
            
            # Get AI narrative - try OLLAMA first, then fallback to comprehensive analysis
            ai_narrative = None
            try:
                if self.ollama_client.check_connection():
                    prompt = self.create_analysis_prompt(symbol, analysis_data, news_data)
                    ai_narrative = self.ollama_client.generate_response(prompt)
            except Exception as e:
                logger.warning(f"OLLAMA connection failed: {e}")
            
            # If OLLAMA failed, generate comprehensive fallback analysis
            if not ai_narrative or "unavailable" in ai_narrative.lower() or "failed" in ai_narrative.lower():
                ai_narrative = self.generate_fallback_analysis(symbol, analysis_data, news_data, ai_results)
            
            return {
                **ai_results,
                'ai_narrative': ai_narrative,
                'analysis_timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing stock {symbol}: {e}")
            return {
                'ai_score': 0,
                'signal_classification': 'Error',
                'component_scores': {},
                'confidence': 0,
                'ai_narrative': f"Analysis failed: {str(e)}",
                'analysis_timestamp': time.time()
            }
    
    def generate_fallback_analysis(self, symbol: str, analysis_data: Dict[str, Any], news_data: str, ai_results: Dict[str, Any]) -> str:
        """Generate comprehensive analysis when OLLAMA is not available"""
        try:
            analysis = []
            
            # Get key metrics
            current_price = analysis_data.get('current_price', 'N/A')
            daily_trend = analysis_data.get('daily_trend', {}).get('overall_trend', 'neutral')
            weekly_trend = analysis_data.get('weekly_trend', {}).get('overall_trend', 'neutral')
            rsi = analysis_data.get('rsi', 50)
            volume_ratio = analysis_data.get('volume_ratio', 1.0)
            ai_score = ai_results.get('ai_score', 0)
            signal = ai_results.get('signal_classification', 'Hold')
            
            # Trend Analysis
            if daily_trend == weekly_trend and daily_trend != 'neutral':
                trend_strength = "strong alignment between daily and weekly trends"
                if daily_trend == 'bullish':
                    analysis.append(f"**Trend Analysis**: {symbol} shows {trend_strength} in a bullish direction, indicating sustained upward momentum.")
                else:
                    analysis.append(f"**Trend Analysis**: {symbol} shows {trend_strength} in a bearish direction, suggesting continued downward pressure.")
            else:
                analysis.append(f"**Trend Analysis**: {symbol} shows mixed signals with daily trend ({daily_trend}) diverging from weekly trend ({weekly_trend}), indicating potential trend transition.")
            
            # Technical Indicators Assessment
            rsi_assessment = ""
            if rsi > 70:
                rsi_assessment = "overbought territory, suggesting potential downward pressure"
            elif rsi < 30:
                rsi_assessment = "oversold conditions, indicating potential buying opportunity"
            else:
                rsi_assessment = "neutral territory, showing balanced momentum"
            
            volume_assessment = ""
            if volume_ratio > 1.5:
                volume_assessment = "significantly above average volume, confirming price movement validity"
            elif volume_ratio < 0.8:
                volume_assessment = "below average volume, suggesting caution in trend confirmation"
            else:
                volume_assessment = "moderate volume levels, supporting current price action"
            
            analysis.append(f"**Technical Indicators**: RSI at {rsi:.1f} indicates {rsi_assessment}. Trading volume is {volume_assessment}.")
            
            # News Impact Assessment
            if news_data and "No recent news" not in news_data:
                news_impact = self._analyze_news_sentiment_fallback(news_data)
                analysis.append(f"**News Analysis**: {news_impact}")
            else:
                analysis.append("**News Analysis**: Limited recent news available. Technical analysis remains primary focus for trading decisions.")
            
            # AI Score Interpretation
            score_interpretation = ""
            if ai_score >= 75:
                score_interpretation = "strong positive signals across multiple indicators"
            elif ai_score >= 60:
                score_interpretation = "moderately positive technical setup"
            elif ai_score >= 40:
                score_interpretation = "mixed signals requiring cautious approach"
            else:
                score_interpretation = "weak technical indicators suggesting defensive positioning"
            
            analysis.append(f"**Overall Assessment**: AI score of {ai_score:.1f} reflects {score_interpretation}. Signal classification: {signal}.")
            
            # Trading Recommendation
            recommendation = self._generate_trading_recommendation(ai_score, daily_trend, weekly_trend, rsi, volume_ratio)
            analysis.append(f"**Trading Recommendation**: {recommendation}")
            
            return " ".join(analysis)
            
        except Exception as e:
            logger.error(f"Error generating fallback analysis: {e}")
            return f"Technical analysis for {symbol}: AI Score {ai_results.get('ai_score', 0):.1f} suggests {ai_results.get('signal_classification', 'Hold')} position. Monitor key technical levels and volume confirmation for entry/exit signals."
    
    def _analyze_news_sentiment_fallback(self, news_data: str) -> str:
        """Simple news sentiment analysis for fallback"""
        try:
            news_lower = news_data.lower()
            
            positive_count = sum(1 for word in ['growth', 'profit', 'beat', 'strong', 'bullish', 'buy', 'upgrade', 'optimistic'] if word in news_lower)
            negative_count = sum(1 for word in ['loss', 'decline', 'bearish', 'sell', 'downgrade', 'pessimistic', 'risk'] if word in news_lower)
            
            if positive_count > negative_count * 1.5:
                return "Recent news sentiment appears positive, with growth-oriented coverage potentially supporting upward price movement."
            elif negative_count > positive_count * 1.5:
                return "Recent news sentiment shows bearish tone, with risk factors potentially weighing on stock performance."
            else:
                return "News sentiment appears balanced, with mixed coverage requiring technical analysis for direction."
                
        except Exception:
            return "News analysis available but requires careful interpretation alongside technical indicators."
    
    def _generate_trading_recommendation(self, ai_score: float, daily_trend: str, weekly_trend: str, rsi: float, volume_ratio: float) -> str:
        """Generate specific trading recommendation"""
        try:
            if ai_score >= 75 and daily_trend == 'bullish' and weekly_trend == 'bullish':
                return "Strong buy signal with aligned trends. Consider position building on volume confirmation."
            elif ai_score >= 60 and daily_trend == 'bullish':
                return "Buy signal with favorable technical setup. Watch for volume confirmation on breakouts."
            elif ai_score <= 25 and daily_trend == 'bearish' and weekly_trend == 'bearish':
                return "Strong sell signal with deteriorating technicals. Consider defensive positioning."
            elif ai_score <= 40 and daily_trend == 'bearish':
                return "Sell signal with weak technical indicators. Monitor support levels for further breakdown."
            elif rsi > 75:
                return "Hold/sell signal due to overbought conditions. Wait for pullback before considering entry."
            elif rsi < 25:
                return "Hold/buy signal due to oversold conditions. Look for reversal signals before entry."
            else:
                return "Hold signal with mixed indicators. Maintain current position and monitor for trend clarification."
                
        except Exception:
            return "Hold signal recommended. Monitor technical developments for trend clarification."