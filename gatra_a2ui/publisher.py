"""
GATRA A2UI Publisher

Handles publishing A2UI schemas to the frontend via Pub/Sub and audit logging.
"""

import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from google.cloud import pubsub_v1
from google.cloud import bigquery


logger = logging.getLogger(__name__)


class A2UIPublisher:
    """
    Publishes A2UI schemas to frontend with audit trail.

    Example:
        publisher = A2UIPublisher(project_id="chronicle-dev-2be9")

        schema = builder.build()
        publisher.publish(
            schema=schema,
            screen_type="attack_narrative",
            incident_id="INC-2025-001",
            agent_id="TAA"
        )
    """

    def __init__(
        self,
        project_id: str,
        topic_name: str = "gatra-a2ui-updates",
        audit_dataset: str = "gatra_audit",
        audit_table: str = "a2ui_states"
    ):
        """
        Initialize A2UI publisher.

        Args:
            project_id: GCP project ID
            topic_name: Pub/Sub topic for A2UI updates
            audit_dataset: BigQuery dataset for audit logs
            audit_table: BigQuery table for A2UI state logs
        """
        self.project_id = project_id
        self.topic_name = topic_name
        self.audit_dataset = audit_dataset
        self.audit_table = audit_table

        # Initialize Pub/Sub publisher
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(project_id, topic_name)

        # Initialize BigQuery client for audit logging
        self.bq_client = bigquery.Client(project=project_id)

    def publish(
        self,
        schema: Dict[str, Any],
        screen_type: str,
        agent_id: str,
        incident_id: Optional[str] = None,
        priority: str = "normal",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Publish A2UI schema to frontend.

        Args:
            schema: A2UI JSON schema
            screen_type: Type of screen (e.g., "attack_narrative", "decision_panel")
            agent_id: Agent that generated the UI (e.g., "ADA", "TAA")
            incident_id: Optional incident ID
            priority: Priority level ("low", "normal", "high", "critical")
            metadata: Additional metadata

        Returns:
            Message ID from Pub/Sub
        """
        # Generate schema hash for audit trail
        schema_json = json.dumps(schema, sort_keys=True)
        schema_hash = hashlib.sha256(schema_json.encode()).hexdigest()

        # Create audit record
        audit_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": agent_id,
            "screen_type": screen_type,
            "incident_id": incident_id,
            "schema_hash": schema_hash,
            "schema": schema_json,
            "priority": priority,
            "metadata": json.dumps(metadata) if metadata else None
        }

        # Log to audit trail (async)
        try:
            self._log_to_audit(audit_record)
        except Exception as e:
            logger.error(f"Failed to log A2UI state to audit trail: {e}")

        # Prepare Pub/Sub message
        message_data = json.dumps(schema).encode("utf-8")

        # Publish attributes for filtering
        attributes = {
            "screen_type": screen_type,
            "agent_id": agent_id,
            "priority": priority,
            "schema_hash": schema_hash
        }

        if incident_id:
            attributes["incident_id"] = incident_id

        if metadata:
            # Add metadata as attributes (flattened)
            for key, value in metadata.items():
                attributes[f"meta_{key}"] = str(value)

        # Publish to Pub/Sub
        try:
            future = self.publisher.publish(
                self.topic_path,
                message_data,
                **attributes
            )
            message_id = future.result(timeout=10)
            logger.info(
                f"Published A2UI update to {self.topic_name} "
                f"(screen_type={screen_type}, agent={agent_id}, message_id={message_id})"
            )
            return message_id

        except Exception as e:
            logger.error(f"Failed to publish A2UI schema: {e}")
            raise

    def _log_to_audit(self, audit_record: Dict[str, Any]):
        """Log A2UI state to BigQuery audit table"""
        table_ref = f"{self.project_id}.{self.audit_dataset}.{self.audit_table}"

        try:
            errors = self.bq_client.insert_rows_json(table_ref, [audit_record])
            if errors:
                logger.error(f"BigQuery audit logging errors: {errors}")
        except Exception as e:
            logger.error(f"Failed to write to audit table: {e}")
            # Don't raise - audit logging failure shouldn't block UI publishing

    def publish_update(
        self,
        component_id: str,
        component_update: Dict[str, Any],
        screen_type: str,
        agent_id: str,
        incident_id: Optional[str] = None
    ) -> str:
        """
        Publish incremental component update (for streaming UIs).

        Args:
            component_id: ID of component to update
            component_update: Updated component definition
            screen_type: Type of screen
            agent_id: Agent making the update
            incident_id: Optional incident ID

        Returns:
            Message ID from Pub/Sub
        """
        # Create partial update schema
        update_schema = {
            "componentUpdate": {
                "id": component_id,
                "component": component_update
            }
        }

        return self.publish(
            schema=update_schema,
            screen_type=screen_type,
            agent_id=agent_id,
            incident_id=incident_id,
            metadata={"update_type": "incremental"}
        )


class StreamingA2UIPublisher(A2UIPublisher):
    """
    Publisher that supports streaming UI updates.

    Useful for long-running agent processes that build UI incrementally.

    Example:
        publisher = StreamingA2UIPublisher(project_id="chronicle-dev-2be9")

        with publisher.stream(screen_type="investigation", agent_id="TAA") as stream:
            stream.add_component(component1)
            # ... agent continues analyzing ...
            stream.add_component(component2)
            # ... more analysis ...
            stream.finalize()
    """

    def stream(self, screen_type: str, agent_id: str, incident_id: Optional[str] = None):
        """Create streaming context"""
        return StreamingContext(self, screen_type, agent_id, incident_id)


class StreamingContext:
    """Context manager for streaming A2UI updates"""

    def __init__(
        self,
        publisher: StreamingA2UIPublisher,
        screen_type: str,
        agent_id: str,
        incident_id: Optional[str] = None
    ):
        self.publisher = publisher
        self.screen_type = screen_type
        self.agent_id = agent_id
        self.incident_id = incident_id
        self.components = []

    def __enter__(self):
        # Send "begin rendering" signal
        self.publisher.publish(
            schema={"beginRendering": {"screen_type": self.screen_type}},
            screen_type=self.screen_type,
            agent_id=self.agent_id,
            incident_id=self.incident_id,
            metadata={"streaming": True}
        )
        return self

    def add_component(self, component: Dict[str, Any]):
        """Add component and publish incrementally"""
        self.components.append(component)

        # Publish incremental update
        self.publisher.publish_update(
            component_id=component["id"],
            component_update=component["component"],
            screen_type=self.screen_type,
            agent_id=self.agent_id,
            incident_id=self.incident_id
        )

    def finalize(self):
        """Send complete UI"""
        final_schema = {
            "surfaceUpdate": {
                "components": self.components
            }
        }

        self.publisher.publish(
            schema=final_schema,
            screen_type=self.screen_type,
            agent_id=self.agent_id,
            incident_id=self.incident_id,
            metadata={"streaming": True, "finalized": True}
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Normal completion - ensure finalized
            if self.components:
                self.finalize()
