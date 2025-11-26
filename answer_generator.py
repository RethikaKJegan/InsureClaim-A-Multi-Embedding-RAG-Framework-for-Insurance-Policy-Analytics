# Week 3-4: Gemini 2.0 Flash Integration with Similarity-Based Confidence
# File: answer_generator.py
# Production-ready version - NO dummy data

import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
import re

# Required installation:
# pip install google-generativeai

import google.generativeai as genai


@dataclass
class Answer:
    """Structured answer with evidence and metadata"""
    question: str
    answer: str
    evidence: List[Dict]
    confidence: float
    confidence_breakdown: Dict[str, float]
    reasoning: str
    sources: List[str]


# ============================================================================
# CONFIDENCE SCORING FUNCTIONS (Similarity-Based)
# ============================================================================

def compute_similarity_confidence(context_chunks: List[Dict]) -> float:
    """
    Compute confidence based on semantic similarity scores from vector search.
    This is MORE RELIABLE than asking the LLM to rate its own confidence.
    
    Args:
        context_chunks: Retrieved chunks with similarity scores
        
    Returns:
        Confidence score (0.0 to 1.0)
    """
    sims = [chunk.get("similarity", 0) for chunk in context_chunks]
    
    if not sims:
        return 0.0
    
    max_sim = max(sims)
    avg_sim = sum(sims) / len(sims)
    min_sim = min(sims)
    
    # Weighted formula (favor top result but consider overall quality)
    base_confidence = 0.7 * max_sim + 0.3 * avg_sim
    
    # Penalty if there's too much variance (inconsistent results)
    variance_penalty = 0
    if len(sims) > 1:
        variance = max_sim - min_sim
        if variance > 0.3:  # Large spread = less confident
            variance_penalty = 0.1
    
    final_confidence = max(0.0, base_confidence - variance_penalty)
    
    return round(final_confidence, 2)


def compute_hybrid_confidence(
    context_chunks: List[Dict], 
    answer_text: str
) -> Dict[str, float]:
    """
    Multi-factor confidence scoring with breakdown
    
    Returns:
        Dictionary with overall confidence and component scores
    """
    # 1. Similarity confidence (50% weight)
    sim_conf = compute_similarity_confidence(context_chunks)
    
    # 2. Coverage confidence (20% weight)
    num_chunks = len(context_chunks)
    coverage_conf = min(num_chunks / 3, 1.0)
    
    # 3. Answer quality check (20% weight)
    uncertainty_phrases = [
        "not found", "unclear", "cannot determine", 
        "insufficient information", "no information available",
        "unable to answer", "not specified", "information not found"
    ]
    
    answer_lower = answer_text.lower()
    has_uncertainty = any(phrase in answer_lower for phrase in uncertainty_phrases)
    
    if has_uncertainty:
        quality_conf = 0.3
    else:
        word_count = len(answer_text.split())
        quality_conf = min(word_count / 50, 1.0)
    
    # 4. Source diversity (10% weight)
    unique_sources = len(set(chunk.get('source_file', '') for chunk in context_chunks))
    diversity_conf = min(unique_sources / 2, 1.0)
    
    # Weighted combination
    weights = {
        'similarity': 0.5,
        'coverage': 0.2,
        'quality': 0.2,
        'diversity': 0.1
    }
    
    overall_confidence = (
        weights['similarity'] * sim_conf +
        weights['coverage'] * coverage_conf +
        weights['quality'] * quality_conf +
        weights['diversity'] * diversity_conf
    )
    
    return {
        'overall': round(overall_confidence, 2),
        'similarity': sim_conf,
        'coverage': round(coverage_conf, 2),
        'quality': round(quality_conf, 2),
        'diversity': round(diversity_conf, 2)
    }


# ============================================================================
# GEMINI ANSWER GENERATOR
# ============================================================================

class GeminiAnswerGenerator:
    """Generate intelligent answers using Gemini 2.0 Flash"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize Gemini API
        
        Args:
            api_key: Google AI API key (or set GOOGLE_API_KEY env var)
            model_name: Gemini model to use
        """
        self.api_key = api_key or os.getenv('AIzaSyBcWKgR2z_ejabRyNZbEJ236Cs8rm5giMI')
        if not self.api_key:
            raise ValueError(
                "Google API key required. Either:\n"
                "1. Pass api_key parameter, or\n"
                "2. Set GOOGLE_API_KEY environment variable\n"
                "Get your key at: https://makersuite.google.com/app/apikey"
            )
        
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        
        print(f"✅ Gemini {model_name} initialized")
    
    def create_prompt(self, question: str, context_chunks: List[Dict]) -> str:
        """
        Create a structured prompt for Gemini
        
        Args:
            question: User's question
            context_chunks: Retrieved document chunks
            
        Returns:
            Formatted prompt string
        """
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = f"[Source {i}: {chunk['source_file']}, Page {chunk.get('page_number', 'N/A')}]"
            text = chunk['text']
            
            sim = chunk.get('similarity')
            if sim is not None:
                source += f" (Relevance: {sim:.2f})"
            
            context_parts.append(f"{source}\n{text}\n")
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are a policy analysis assistant. Based on the provided document context, answer the user's question accurately and concisely.

CONTEXT FROM POLICY DOCUMENTS:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Provide a clear, direct answer (2-3 sentences)
2. Quote relevant evidence from the context (cite source numbers)
3. Explain your reasoning briefly
4. If the context doesn't contain the answer, say "Information not found in provided documents"

RESPONSE FORMAT (use exactly this structure):
ANSWER: [Your 2-3 sentence answer]

EVIDENCE: [Quote relevant parts with source numbers like [Source 1]]

REASONING: [Brief explanation of why you gave this answer]

Now answer the question:"""
        
        return prompt
    
    def parse_response(self, response_text: str, question: str, context_chunks: List[Dict]) -> Dict:
        """
        Parse Gemini's response into structured components
        
        Args:
            response_text: Raw response from Gemini
            question: Original question
            context_chunks: Context used for answer
            
        Returns:
            Dictionary with parsed components
        """
        answer_match = re.search(r'ANSWER:\s*(.*?)(?=EVIDENCE:|REASONING:|$)', response_text, re.DOTALL)
        evidence_match = re.search(r'EVIDENCE:\s*(.*?)(?=REASONING:|$)', response_text, re.DOTALL)
        reasoning_match = re.search(r'REASONING:\s*(.*?)$', response_text, re.DOTALL)
        
        answer_text = answer_match.group(1).strip() if answer_match else "Unable to generate answer"
        evidence_text = evidence_match.group(1).strip() if evidence_match else ""
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        
        evidence = []
        for chunk in context_chunks:
            evidence.append({
                'source_file': chunk['source_file'],
                'page_number': chunk.get('page_number'),
                'text': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                'similarity': chunk.get('similarity')
            })
        
        sources = list(set(chunk['source_file'] for chunk in context_chunks))
        
        return {
            'answer': answer_text,
            'evidence': evidence,
            'reasoning': reasoning,
            'sources': sources
        }
    
    def generate_answer(
        self, 
        question: str, 
        context_chunks: List[Dict],
        temperature: float = 0.3,
        use_hybrid_confidence: bool = True
    ) -> Answer:
        """
        Generate an answer to the question using retrieved context
        
        Args:
            question: User's question
            context_chunks: Retrieved chunks (MUST include 'similarity' scores)
            temperature: LLM temperature (0.0-1.0, lower = more focused)
            use_hybrid_confidence: Use multi-factor confidence (recommended)
            
        Returns:
            Structured Answer object with similarity-based confidence
        """
        if not context_chunks:
            return Answer(
                question=question,
                answer="No relevant information found in the documents.",
                evidence=[],
                confidence=0.0,
                confidence_breakdown={'overall': 0.0},
                reasoning="No context provided",
                sources=[]
            )
        
        prompt = self.create_prompt(question, context_chunks)
        
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=1000,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            parsed = self.parse_response(response.text, question, context_chunks)
            
            # IMPROVED: Compute confidence from similarity scores (NOT from LLM)
            if use_hybrid_confidence:
                conf_breakdown = compute_hybrid_confidence(context_chunks, parsed['answer'])
                final_confidence = conf_breakdown['overall']
            else:
                final_confidence = compute_similarity_confidence(context_chunks)
                conf_breakdown = {
                    'overall': final_confidence,
                    'similarity': final_confidence,
                    'coverage': 0.0,
                    'quality': 0.0,
                    'diversity': 0.0
                }
            
            return Answer(
                question=question,
                answer=parsed['answer'],
                evidence=parsed['evidence'],
                confidence=final_confidence,
                confidence_breakdown=conf_breakdown,
                reasoning=parsed['reasoning'],
                sources=parsed['sources']
            )
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return Answer(
                question=question,
                answer=f"Error generating answer: {str(e)}",
                evidence=[],
                confidence=0.0,
                confidence_breakdown={'overall': 0.0},
                reasoning="Generation failed",
                sources=[]
            )
    
    def generate_summary(self, document_chunks: List[Dict], max_chunks: int = 10) -> str:
        """
        Generate an executive summary of the document
        
        Args:
            document_chunks: All chunks from a document
            max_chunks: Maximum chunks to include in summary
            
        Returns:
            Summary text
        """
        chunks_to_use = document_chunks[:max_chunks]
        context = "\n\n".join(chunk['text'] for chunk in chunks_to_use)
        
        prompt = f"""Provide a concise executive summary of this policy document. Include:
1. Main purpose of the policy
2. Key coverage areas (3-5 bullet points)
3. Important limitations or exclusions

DOCUMENT CONTENT:
{context}

EXECUTIVE SUMMARY:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating summary: {str(e)}"


# ============================================================================
# INTELLIGENT Q&A SYSTEM
# ============================================================================

class IntelligentQASystem:
    """Complete Q&A system with semantic search + Gemini answers"""
    
    def __init__(self, search_engine, gemini_api_key: Optional[str] = None):
        """
        Initialize intelligent Q&A system
        
        Args:
            search_engine: SemanticSearchEngine from Week 2
            gemini_api_key: Google AI API key
        """
        self.search_engine = search_engine
        self.answer_generator = GeminiAnswerGenerator(api_key=gemini_api_key)
    
    def ask(
        self, 
        question: str, 
        top_k: int = 5, 
        temperature: float = 0.3,
        use_hybrid_confidence: bool = True
    ) -> Answer:
        """
        Ask a question and get an intelligent answer
        
        Args:
            question: User's question
            top_k: Number of context chunks to retrieve
            temperature: LLM creativity (0.0-1.0)
            use_hybrid_confidence: Use multi-factor confidence scoring
            
        Returns:
            Structured Answer object with similarity-based confidence
        """
        # Step 1: Semantic search for relevant chunks
        search_results = self.search_engine.search(question, k=top_k)
        
        # Step 2: Generate answer using Gemini with similarity-based confidence
        answer = self.answer_generator.generate_answer(
            question, 
            search_results,
            temperature=temperature,
            use_hybrid_confidence=use_hybrid_confidence
        )
        
        return answer
    
    def display_answer(self, answer: Answer, show_breakdown: bool = True):
        """Pretty print an answer with confidence breakdown"""
        print("\n" + "="*60)
        print(f"❓ QUESTION: {answer.question}")
        print("="*60)
        
        print(f"\n✅ ANSWER:")
        print(f"   {answer.answer}")
        
        conf_pct = answer.confidence * 100
        if conf_pct >= 70:
            emoji = "🟢"
        elif conf_pct >= 40:
            emoji = "🟡"
        else:
            emoji = "🔴"
        
        print(f"\n{emoji} CONFIDENCE: {conf_pct:.0f}%")
        
        if show_breakdown and 'similarity' in answer.confidence_breakdown:
            breakdown = answer.confidence_breakdown
            print(f"   └─ Similarity: {breakdown['similarity']:.0%}")
            if breakdown.get('coverage', 0) > 0:
                print(f"   └─ Coverage: {breakdown['coverage']:.0%}")
                print(f"   └─ Quality: {breakdown['quality']:.0%}")
                print(f"   └─ Diversity: {breakdown['diversity']:.0%}")
        
        if answer.evidence:
            print(f"\n📄 EVIDENCE:")
            for i, ev in enumerate(answer.evidence[:3], 1):
                print(f"   [{i}] {ev['source_file']} (Page {ev.get('page_number', 'N/A')})")
                if ev.get('similarity'):
                    print(f"       Similarity: {ev['similarity']:.3f}")
                print(f"       \"{ev['text']}\"")
        
        print(f"\n💭 REASONING:")
        print(f"   {answer.reasoning}")
        
        print(f"\n📚 SOURCES: {', '.join(answer.sources)}")
        print("="*60)


# ============================================================================
# NO DEMO CODE - PRODUCTION READY
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("📄 answer_generator.py - Production Module")
    print("="*60)
    print("\nThis module is ready to be imported into your main application.")
    print("\nUsage:")
    print("------")
    print("from answer_generator import IntelligentQASystem")
    print("from vector_store import SemanticSearchEngine")
    print("")
    print("# Initialize")
    print("search_engine = SemanticSearchEngine()")
    print("qa_system = IntelligentQASystem(search_engine, gemini_api_key='your-key')")
    print("")
    print("# Ask questions")
    print("answer = qa_system.ask('Your question?')")
    print("qa_system.display_answer(answer)")
    print("\n" + "="*60)
    print("✅ Module loaded successfully - ready for integration!")
    print("="*60 + "\n")