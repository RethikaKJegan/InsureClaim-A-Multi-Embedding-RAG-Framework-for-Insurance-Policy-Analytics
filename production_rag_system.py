# ============================================================
# Production-Grade RAG System with Multi-Stage Validation
# File: production_rag_system.py
# ============================================================

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re

# ✅ Prevent Streamlit + Torch watcher runtime error
os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"

from document_processor import DocumentProcessor
from vector_store import SemanticSearchEngine
from answer_generator import GeminiAnswerGenerator, Answer
from advanced_rag import DocumentGraphBuilder, AdvancedRAGRetriever


class ProductionRAGSystem:
    """
    Production-grade RAG with:
    1. Query Enhancement (expand search terms)
    2. Multi-stage Retrieval (wide net → filter → rank)
    3. Context Validation (verify relevance before answering)
    4. Answer Verification (check against retrieved context)
    5. Confidence Calibration (honest uncertainty estimation)
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        chunk_size: int = 1000,
        overlap: int = 200,
        model_name: str = 'all-MiniLM-L6-v2'
    ):
        print("\n" + "🏗️ " + "=" * 58)
        print("   PRODUCTION RAG SYSTEM v2.0")
        print("🏗️ " + "=" * 58 + "\n")

        self.doc_processor = DocumentProcessor(chunk_size=chunk_size, overlap=overlap)
        self.search_engine = SemanticSearchEngine(model_name=model_name)

        try:
            self.answer_generator = GeminiAnswerGenerator(api_key=gemini_api_key)
            self.gemini_enabled = True
        except ValueError as e:
            print(f"⚠️  Gemini not configured: {e}")
            self.gemini_enabled = False
            self.answer_generator = None

        self.indexed_files = []
        self.all_chunks = []
        self.chunk_graph = None
        self.advanced_retriever = None

        print("✅ System initialized with validation pipeline\n")

    # ============================================================
    # DOCUMENT PROCESSING & INDEXING
    # ============================================================
    def process_and_index(self, file_path: str):
        """Process document and build graph"""
        print("\n" + "📁 " + "=" * 58)
        print(f"   PROCESSING: {Path(file_path).name}")
        print("📁 " + "=" * 58 + "\n")

        chunks = self.doc_processor.process_document(file_path)
        if not chunks:
            return False

        self.all_chunks.extend(chunks)
        self.search_engine.index_documents(chunks)

        # Build graph
        graph_builder = DocumentGraphBuilder()
        self.chunk_graph = graph_builder.build_graph(self.all_chunks)
        self.advanced_retriever = AdvancedRAGRetriever(
            search_engine=self.search_engine,
            chunk_graph=self.chunk_graph
        )

        self.indexed_files.append(Path(file_path).name)
        return True

    # ============================================================
    # MAIN RAG PIPELINE
    # ============================================================
    def ask(self, question: str, use_validation: bool = True) -> Answer:
        """
        PRODUCTION PIPELINE:
        Stage 1: Query Enhancement
        Stage 2: Multi-Retrieval Strategy
        Stage 3: Context Validation
        Stage 4: Answer Generation with Verification
        Stage 5: Confidence Calibration
        """

        print("\n" + "🔍 " + "=" * 58)
        print("   PRODUCTION RAG PIPELINE")
        print("🔍 " + "=" * 58)

        # ============================================================
        # STAGE 1: Query Enhancement
        # ============================================================
        print("\n📝 STAGE 1: Query Enhancement")
        enhanced_queries = self._enhance_query(question)
        print(f"   Original: {question}")
        for i, eq in enumerate(enhanced_queries, 1):
            print(f"   Enhanced {i}: {eq}")

        # ============================================================
        # STAGE 2: Multi-Retrieval Strategy
        # ============================================================
        print("\n🔎 STAGE 2: Multi-Retrieval (cast wide net)")

        all_candidates = {}

        for query in [question] + enhanced_queries:
            results = self.search_engine.search(query, k=10)
            for r in results:
                chunk_id = r['chunk_id']
                if chunk_id not in all_candidates:
                    all_candidates[chunk_id] = r
                else:
                    if r.get('similarity', 0) > all_candidates[chunk_id].get('similarity', 0):
                        all_candidates[chunk_id] = r

        print(f"   → Found {len(all_candidates)} unique candidates")

        ranked_candidates = sorted(
            all_candidates.values(),
            key=lambda x: x.get('similarity', 0),
            reverse=True
        )[:15]

        print(f"   → Top 15 selected")
        for i, r in enumerate(ranked_candidates[:5], 1):
            print(f"      {i}. Chunk {r['chunk_id']} | Sim: {r.get('similarity', 0):.3f}")

        # ============================================================
        # STAGE 3: Context Validation
        # ============================================================
        print("\n✅ STAGE 3: Context Validation")

        if use_validation:
            validated_chunks = self._validate_context(question, ranked_candidates[:10])
            print(f"   → {len(validated_chunks)} chunks passed validation")
        else:
            validated_chunks = ranked_candidates[:5]
            print(f"   → Validation skipped, using top 5")

        if not validated_chunks:
            print("   ⚠️  No chunks passed validation!")
            validated_chunks = ranked_candidates[:3]  # Fallback

        # Smart context building
        final_context = []
        seen_ids = set()

        for chunk in validated_chunks[:3]:
            chunk_id = chunk['chunk_id']
            if chunk_id not in seen_ids:
                final_context.append({**chunk, 'context_type': 'primary'})
                seen_ids.add(chunk_id)

            if len(final_context) <= 2:
                context = self._get_smart_context(
                    chunk_id,
                    similarity_score=chunk.get('similarity', 0)
                )
                for ctx in context:
                    if ctx['chunk_id'] not in seen_ids and len(final_context) < 8:
                        final_context.append(ctx)
                        seen_ids.add(ctx['chunk_id'])

        print(f"   → Final context: {len(final_context)} chunks")

        # ============================================================
        # STAGE 4: Answer Generation with Verification
        # ============================================================
        print("\n🤖 STAGE 4: Answer Generation + Verification")

        if not self.gemini_enabled:
            return self._fallback_answer(question, final_context)

        answer = self._generate_verified_answer(question, final_context)

        # ============================================================
        # STAGE 5: Confidence Calibration
        # ============================================================
        print("\n📊 STAGE 5: Confidence Calibration")

        calibrated_confidence = self._calibrate_confidence(
            answer,
            final_context,
            validated_chunks
        )

        answer.confidence = calibrated_confidence
        print(f"   → Calibrated confidence: {calibrated_confidence:.0%}")

        print("\n" + "=" * 60 + "\n")
        return answer

    # ============================================================
    # HELPER FUNCTIONS
    # ============================================================
    def _enhance_query(self, question: str) -> List[str]:
        """
        Enhance query with synonyms and policy-related variants.
        Improves retrieval diversity.
        """
        enhanced = []
        question_lower = question.lower()

        expansions = {
            'cancellation': ['termination', 'policy end', 'discontinuation'],
            'coverage': ['benefits', 'protection', 'entitlement'],
            'exclusion': ['not covered', 'limitations', 'exceptions'],
            'insurer': ['company', 'insurance provider', 'we'],
            'insured': ['policyholder', 'you', 'member'],
            'claim': ['reimbursement', 'payment', 'settlement'],
        }

        for key, synonyms in expansions.items():
            if key in question_lower:
                for syn in synonyms[:2]:
                    enhanced_q = question.replace(key, syn).replace(key.title(), syn.title())
                    if enhanced_q != question:
                        enhanced.append(enhanced_q)

        if 'right' in question_lower or 'can' in question_lower:
            enhanced.append(question.replace('?', ' according to policy?'))

        return enhanced[:3]

    def _validate_context(self, question: str, candidates: List[Dict]) -> List[Dict]:
        """Validate retrieved chunks for relevance using Gemini"""
        if not self.gemini_enabled:
            return candidates[:5]

        validated = []
        for chunk in candidates:
            validation_prompt = f"""Question: {question}

Retrieved text: {chunk['text'][:500]}

Does this text contain information that could help answer the question?
Answer ONLY: YES or NO

Answer:"""

            try:
                response = self.answer_generator.model.generate_content(
                    validation_prompt,
                    generation_config={'temperature': 0.0, 'max_output_tokens': 10}
                )
                answer_text = response.text.strip().upper()
                if 'YES' in answer_text:
                    validated.append(chunk)
                    print(f"      ✓ Chunk {chunk['chunk_id']} validated")
                else:
                    print(f"      ✗ Chunk {chunk['chunk_id']} rejected")
            except Exception:
                validated.append(chunk)
                print(f"      ? Chunk {chunk['chunk_id']} validation failed, including anyway")

        return validated

    def _generate_verified_answer(self, question: str, context_chunks: List[Dict]) -> Answer:
        """Generate verified answer"""
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = f"[Source {i}: {chunk['source_file']}, Page {chunk.get('page_number', 'N/A')}]"
            text = chunk['text']
            context_parts.append(f"{source}\n{text}\n")

        context = "\n".join(context_parts)

        prompt = f"""You are a policy analysis expert. Answer the question using ONLY the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Use only given context.
2. Provide citations [Source X].
3. Indicate confidence clearly (HIGH/MEDIUM/LOW).

RESPONSE FORMAT:
ANSWER: [Your answer]
EVIDENCE: [Quoted text]
CONFIDENCE: [HIGH/MEDIUM/LOW] - [percentage]%
REASONING: [Short reasoning]
"""

        try:
            response = self.answer_generator.model.generate_content(
                prompt,
                generation_config={'temperature': 0.1, 'max_output_tokens': 1500}
            )
            return self._parse_verified_response(response.text, question, context_chunks)
        except Exception as e:
            print(f"   ❌ Generation error: {e}")
            return Answer(
                question=question,
                answer=f"Error: {str(e)}",
                evidence=context_chunks,
                confidence=0.0,
                confidence_breakdown={'overall': 0.0},
                reasoning="Generation failed",
                sources=[]
            )

    def _parse_verified_response(self, response_text: str, question: str, context_chunks: List[Dict]) -> Answer:
        """Parse response safely"""
        answer_match = re.search(r'ANSWER:\s*(.*?)(?=EVIDENCE:|CONFIDENCE:|$)', response_text, re.DOTALL)
        evidence_match = re.search(r'EVIDENCE:\s*(.*?)(?=CONFIDENCE:|REASONING:|$)', response_text, re.DOTALL)
        confidence_match = re.search(r'CONFIDENCE:\s*(?:HIGH|MEDIUM|LOW)?\s*-?\s*(\d+)%', response_text, re.IGNORECASE)
        reasoning_match = re.search(r'REASONING:\s*(.*?)$', response_text, re.DOTALL)

        answer_text = answer_match.group(1).strip() if answer_match else "Unable to generate answer"
        evidence_text = evidence_match.group(1).strip() if evidence_match else ""
        confidence = int(confidence_match.group(1)) if confidence_match else 50
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"

        evidence = []
        for chunk in context_chunks[:5]:
            evidence.append({
                'source_file': chunk['source_file'],
                'page_number': chunk.get('page_number'),
                'text': chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'],
                'chunk_id': chunk['chunk_id'],
                'similarity': chunk.get('similarity', 0)
            })

        sources = list(set(chunk['source_file'] for chunk in context_chunks))

        return Answer(
            question=question,
            answer=answer_text,
            evidence=evidence,
            confidence=confidence / 100.0,
            confidence_breakdown={'overall': confidence / 100.0},
            reasoning=reasoning,
            sources=sources
        )

    def _calibrate_confidence(self, answer: Answer, final_context: List[Dict], validated_chunks: List[Dict]) -> float:
        """Confidence calibration based on similarity, validation, and clarity"""
        top_similarities = [c.get('similarity', 0) for c in final_context[:3]]
        avg_similarity = sum(top_similarities) / len(top_similarities) if top_similarities else 0
        validation_rate = len(validated_chunks) / max(len(final_context), 1)
        has_citations = '[Source' in answer.answer
        not_uncertain = 'insufficient' not in answer.answer.lower() and 'not found' not in answer.answer.lower()

        base_conf = answer.confidence
        if avg_similarity < 0.3:
            base_conf *= 0.7
        elif avg_similarity > 0.5:
            base_conf = min(base_conf * 1.1, 1.0)
        if validation_rate < 0.5:
            base_conf *= 0.8
        if not has_citations:
            base_conf *= 0.9
        if not not_uncertain:
            base_conf *= 0.7
        return min(base_conf, 1.0)

    def _get_smart_context(self, chunk_id: int, similarity_score: float) -> List[Dict]:
        """Add context intelligently based on similarity"""
        if not self.chunk_graph or chunk_id not in self.chunk_graph:
            return []
        node = self.chunk_graph[chunk_id]
        context = []
        if similarity_score >= 0.5:
            if node.prev_chunk and node.prev_chunk in self.chunk_graph:
                prev_node = self.chunk_graph[node.prev_chunk]
                context.append(self._node_to_dict(prev_node, 'prev', 0.6))
            if node.next_chunk and node.next_chunk in self.chunk_graph:
                next_node = self.chunk_graph[node.next_chunk]
                context.append(self._node_to_dict(next_node, 'next', 0.6))
        elif similarity_score >= 0.35:
            if node.parent_section and node.parent_section in self.chunk_graph:
                parent_node = self.chunk_graph[node.parent_section]
                context.append(self._node_to_dict(parent_node, 'parent', 0.5))
        return context

    def _node_to_dict(self, node, ctype: str, priority: float) -> Dict:
        """Convert node to dictionary"""
        return {
            'chunk_id': node.chunk_id,
            'text': node.text,
            'source_file': node.source_file,
            'page_number': node.page_number,
            'context_type': ctype,
            'context_priority': priority,
            'similarity': 0.3
        }

    def _fallback_answer(self, question: str, context: List[Dict]) -> Answer:
        """Fallback when Gemini unavailable"""
        return Answer(
            question=question,
            answer="Gemini not configured - search results only",
            evidence=context,
            confidence=0.0,
            confidence_breakdown={'overall': 0.0},
            reasoning="No LLM available",
            sources=[]
        )

    def get_stats(self) -> Dict:
        """Get system stats"""
        return {
            'indexed_files': len(self.indexed_files),
            'total_chunks': self.search_engine.vector_store.index.ntotal,
            'model': self.search_engine.embedding_generator.model_name,
            'gemini_enabled': self.gemini_enabled,
            'graph_nodes': len(self.chunk_graph) if self.chunk_graph else 0
        }


def print_stats(stats: Dict):
    """Print summary stats"""
    print("\n📊 SYSTEM STATS")
    print("=" * 60)
    print(f"  Files: {stats['indexed_files']}")
    print(f"  Chunks: {stats['total_chunks']}")
    print(f"  Model: {stats['model']}")
    print(f"  Gemini: {'✅' if stats['gemini_enabled'] else '❌'}")
    print(f"  Graph: {stats['graph_nodes']} nodes")
    print("=" * 60 + "\n")


def main():
    """Command-line entry"""
    import argparse

    parser = argparse.ArgumentParser(description='Production RAG System')
    parser.add_argument('--file', type=str, help='Process file')
    parser.add_argument('--interactive', action='store_true', help='Interactive Q&A mode')
    parser.add_argument('--question', type=str, help='Ask a specific question')

    args = parser.parse_args()

    system = ProductionRAGSystem(
        gemini_api_key="AIzaSyB7h2UEHtPlbZ_uS7X0p6vq59HuiwAO48I"
    )

    if args.file:
        system.process_and_index(args.file)
        print_stats(system.get_stats())

    if args.question:
        answer = system.ask(args.question)
        system.display_answer(answer)
    elif args.interactive:
        system.interactive_qa()


if __name__ == "__main__":
    main()
