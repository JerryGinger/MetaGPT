#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : YYYY/MM/DD HH:MM:SS
@Author  : Your Name
@File    : query_rag.py
"""

from __future__ import annotations

from typing import Optional

from metagpt.actions.action import Action
from metagpt.schema import ActionOutput, Message  # Or ActionOutput if preferred
from metagpt.utils.common import OutputParser

try:
    from metagpt.rag.engines.simple import SimpleEngine
except ImportError:
    SimpleEngine = None


class QueryRAG(Action):
    """
    Action to query a RAG (Retrieval Augmented Generation) engine.
    """

    name: str = "QueryRAG"
    query: str = "" # Can be set at initialization or passed to run

    async def run(self, query: Optional[str] = None) -> ActionOutput | Message:
        """
        Runs the RAG query.

        Args:
            query: The query string to be passed to the RAG engine.
                   If None, uses the query set at initialization.

        Returns:
            An ActionOutput or Message containing the RAG engine's response,
            or a message indicating the RAG engine is not available.
        """
        current_query = query if query is not None else self.query
        if not current_query:
            return ActionOutput(content="Error: No query provided for RAG.")

        # Attempt to access rag_engine via self.context.rc (RoleContext)
        # self.context is typically the Role instance when an action is bound to a role.
        rag_engine_found = False
        if SimpleEngine and hasattr(self.context, 'rc') and hasattr(self.context.rc, 'rag_engine'):
            if isinstance(self.context.rc.rag_engine, SimpleEngine):
                rag_engine_found = True
                try:
                    rag_output = await self.context.rc.rag_engine.aquery(current_query)
                    # Consider returning as ActionOutput for consistency if other actions do so
                    # For now, returning a Message object might be more aligned with chat history
                    return Message(content=rag_output, role="assistant", sent_from=self.name)
                except Exception as e:
                    return Message(content=f"Error querying RAG engine: {str(e)}", role="assistant", sent_from=self.name)

        if not rag_engine_found:
            return Message(content="RAG engine not available or not configured correctly.", role="assistant", sent_from=self.name)

        # Fallback if somehow the logic above was bypassed without returning
        return Message(content="RAG query processed with no specific output.", role="assistant", sent_from=self.name)

# Example Usage (for testing purposes, not part of the library code)
# if __name__ == '__main__':
#     import asyncio
#
#     class MockRoleContext:
#         def __init__(self, engine=None):
#             self.rag_engine = engine
#
#     class MockRole:
#         def __init__(self, rc):
#             self.rc = rc
#             self.name = "MockRoleForQueryRAG" # Add name attribute
#
#     class MockSimpleEngine:
#         async def aquery(self, query: str) -> str:
#             return f"Mock RAG response to: {query}"
#
#     async def main():
#         # Test case 1: RAG engine available
#         mock_engine = MockSimpleEngine()
#         mock_rc = MockRoleContext(engine=mock_engine)
#         mock_role = MockRole(rc=mock_rc)
#
#         action1 = QueryRAG()
#         action1.context = mock_role # Manually set context for test
#
#         output1 = await action1.run(query="What is the capital of France?")
#         print(f"Test 1 Output: {output1.content}")
#
#         # Test case 2: RAG engine not available
#         mock_rc_no_engine = MockRoleContext(engine=None)
#         mock_role_no_engine = MockRole(rc=mock_rc_no_engine)
#
#         action2 = QueryRAG()
#         action2.context = mock_role_no_engine # Manually set context
#         output2 = await action2.run(query="Tell me a joke.")
#         print(f"Test 2 Output: {output2.content}")
#
#         # Test case 3: No query provided
#         action3 = QueryRAG()
#         action3.context = mock_role # Manually set context
#         output3 = await action3.run()
#         print(f"Test 3 Output: {output3.content}")
#
#     asyncio.run(main())
