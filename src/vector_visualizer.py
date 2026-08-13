import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Ensure UTF-8 output encoding for Windows terminal
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class VectorVisualizer:
    """Helper module for interactive 2D/3D visualization of dense vector embeddings."""

    def __init__(self, vectors: np.ndarray, metadata: List[Dict[str, Any]], n_clusters: int = 5):
        self.vectors = vectors.astype(np.float32)
        self.metadata = metadata
        self.n_samples, self.n_dim = self.vectors.shape
        self.n_clusters = min(n_clusters, self.n_samples)

        # Fit KMeans for semantic grouping & color coding
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(self.vectors)

        # Fit PCA for 2D & 3D transformations
        self.pca_3d = PCA(n_components=3, random_state=42)
        self.coords_3d_pca = self.pca_3d.fit_transform(self.vectors)

        self.pca_2d = PCA(n_components=2, random_state=42)
        self.coords_2d_pca = self.pca_2d.fit_transform(self.vectors)

        # Precompute t-SNE coordinates
        perplexity_val = min(30, max(5, self.n_samples - 1))
        self.tsne_3d = TSNE(n_components=3, random_state=42, perplexity=perplexity_val, max_iter=800)
        self.coords_3d_tsne = self.tsne_3d.fit_transform(self.vectors)

        self.tsne_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, max_iter=800)
        self.coords_2d_tsne = self.tsne_2d.fit_transform(self.vectors)

        # Generate cluster labels/names
        self.cluster_names = [f"Nhóm Ngữ nghĩa #{i + 1}" for i in range(self.n_clusters)]

    def _get_hover_texts(self) -> List[str]:
        hover_texts = []
        for i, meta in enumerate(self.metadata):
            header = meta.get("header", f"Vector #{i}")
            source = meta.get("source", "Tài liệu")
            loc = meta.get("location", "")
            raw_text = meta.get("text", "")
            snippet = (raw_text[:140] + "...") if len(raw_text) > 140 else raw_text
            # Format HTML hover text for Plotly
            clean_snippet = snippet.replace("\n", "<br>")
            ht = (
                f"<b>{header}</b><br>"
                f"<i>ID: #{i} | Nhóm: {self.cluster_names[self.cluster_labels[i]]}</i><br>"
                f"📌 <b>Vị trí:</b> {loc}<br>"
                f"📄 <b>Nguồn:</b> {source}<br><br>"
                f"<b>Nội dung trích đoạn:</b><br>{clean_snippet}"
            )
            hover_texts.append(ht)
        return hover_texts

    def plot_3d_space(
        self,
        method: str = "PCA",
        query_vector: Optional[np.ndarray] = None,
        query_text: str = "",
        nearest_indices: Optional[List[int]] = None,
    ) -> go.Figure:
        """Create interactive 3D Scatter Plot."""
        coords = self.coords_3d_pca if method == "PCA" else self.coords_3d_tsne
        hover_texts = self._get_hover_texts()

        fig = go.Figure()

        # Add data nodes colored by cluster
        for c in range(self.n_clusters):
            mask = self.cluster_labels == c
            cluster_name = self.cluster_names[c]
            fig.add_trace(
                go.Scatter3d(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    z=coords[mask, 2],
                    mode="markers+text",
                    name=cluster_name,
                    text=[f"#{idx}" for idx in np.where(mask)[0]],
                    textposition="top center",
                    textfont=dict(size=9, color="#64748B"),
                    hovertext=[hover_texts[idx] for idx in np.where(mask)[0]],
                    hoverinfo="text",
                    marker=dict(
                        size=6,
                        opacity=0.88,
                        line=dict(width=1, color="white"),
                    ),
                )
            )

        # If query vector is provided, project and draw connections
        if query_vector is not None:
            if method == "PCA":
                query_coords = self.pca_3d.transform(query_vector.reshape(1, -1))[0]
            else:
                # For t-SNE, use nearest neighbor approximation in 3D
                query_coords = coords[nearest_indices[0]] + np.array([0.05, 0.05, 0.05]) if nearest_indices else np.array([0, 0, 0])

            # Add Query Node (Star)
            fig.add_trace(
                go.Scatter3d(
                    x=[query_coords[0]],
                    y=[query_coords[1]],
                    z=[query_coords[2]],
                    mode="markers+text",
                    name="🎯 Vị trí Câu hỏi (Query)",
                    text=["🔍 CÂU HỎI"],
                    textposition="bottom center",
                    textfont=dict(size=12, color="#EF4444"),
                    hovertext=f"<b>🎯 Câu hỏi người dùng:</b><br>\"{query_text}\"",
                    hoverinfo="text",
                    marker=dict(
                        size=12,
                        symbol="diamond",
                        color="#EF4444",
                        line=dict(width=2, color="#B91C1C"),
                    ),
                )
            )

            # Draw lines from query to nearest neighbor vectors
            if nearest_indices:
                for rank, n_idx in enumerate(nearest_indices, 1):
                    target_coords = coords[n_idx]
                    fig.add_trace(
                        go.Scatter3d(
                            x=[query_coords[0], target_coords[0]],
                            y=[query_coords[1], target_coords[1]],
                            z=[query_coords[2], target_coords[2]],
                            mode="lines",
                            name=f"Top {rank} Match (#{n_idx})",
                            line=dict(color="#F59E0B", width=4, dash="dot"),
                            hoverinfo="none",
                            showlegend=False,
                        )
                    )

        title_method = "PCA (Principal Component Analysis)" if method == "PCA" else "t-SNE"
        fig.update_layout(
            title=dict(
                text=f"🌌 <b>Bản đồ Không gian Vector 3D ({title_method})</b> - {self.n_samples} Điều luật",
                font=dict(size=18, color="#1E3A8A"),
            ),
            scene=dict(
                xaxis=dict(title="Trục Dim 1", gridcolor="#E2E8F0", backgroundcolor="#F8FAFC"),
                yaxis=dict(title="Trục Dim 2", gridcolor="#E2E8F0", backgroundcolor="#F8FAFC"),
                zaxis=dict(title="Trục Dim 3", gridcolor="#E2E8F0", backgroundcolor="#F8FAFC"),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.8)",
            ),
            height=680,
        )
        return fig

    def plot_2d_space(
        self,
        method: str = "PCA",
        query_vector: Optional[np.ndarray] = None,
        query_text: str = "",
        nearest_indices: Optional[List[int]] = None,
    ) -> go.Figure:
        """Create interactive 2D Scatter Plot."""
        coords = self.coords_2d_pca if method == "PCA" else self.coords_2d_tsne
        hover_texts = self._get_hover_texts()

        fig = go.Figure()

        # Add data nodes colored by cluster
        for c in range(self.n_clusters):
            mask = self.cluster_labels == c
            cluster_name = self.cluster_names[c]
            fig.add_trace(
                go.Scatter(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    mode="markers+text",
                    name=cluster_name,
                    text=[f"#{idx}" for idx in np.where(mask)[0]],
                    textposition="top center",
                    textfont=dict(size=9, color="#64748B"),
                    hovertext=[hover_texts[idx] for idx in np.where(mask)[0]],
                    hoverinfo="text",
                    marker=dict(
                        size=10,
                        opacity=0.85,
                        line=dict(width=1, color="white"),
                    ),
                )
            )

        # If query vector is provided, project onto 2D
        if query_vector is not None:
            if method == "PCA":
                query_coords = self.pca_2d.transform(query_vector.reshape(1, -1))[0]
            else:
                query_coords = coords[nearest_indices[0]] + np.array([0.05, 0.05]) if nearest_indices else np.array([0, 0])

            fig.add_trace(
                go.Scatter(
                    x=[query_coords[0]],
                    y=[query_coords[1]],
                    mode="markers+text",
                    name="🎯 Vị trí Câu hỏi (Query)",
                    text=["🔍 CÂU HỎI"],
                    textposition="bottom center",
                    textfont=dict(size=12, color="#EF4444", family="Arial Black"),
                    hovertext=f"<b>🎯 Câu hỏi:</b><br>\"{query_text}\"",
                    hoverinfo="text",
                    marker=dict(
                        size=16,
                        symbol="star",
                        color="#EF4444",
                        line=dict(width=2, color="#7F1D1D"),
                    ),
                )
            )

            # Draw lines to nearest neighbor vectors
            if nearest_indices:
                for rank, n_idx in enumerate(nearest_indices, 1):
                    target_coords = coords[n_idx]
                    fig.add_trace(
                        go.Scatter(
                            x=[query_coords[0], target_coords[0]],
                            y=[query_coords[1], target_coords[1]],
                            mode="lines",
                            name=f"Top {rank} Match (#{n_idx})",
                            line=dict(color="#F59E0B", width=2, dash="dash"),
                            hoverinfo="none",
                            showlegend=False,
                        )
                    )

        title_method = "PCA" if method == "PCA" else "t-SNE"
        fig.update_layout(
            title=dict(
                text=f"🗺️ <b>Bản đồ Không gian Vector 2D ({title_method})</b> - {self.n_samples} Điều luật",
                font=dict(size=18, color="#1E3A8A"),
            ),
            xaxis=dict(title="Trục Dim 1", gridcolor="#E2E8F0", zeroline=False),
            yaxis=dict(title="Trục Dim 2", gridcolor="#E2E8F0", zeroline=False),
            plot_bgcolor="#F8FAFC",
            margin=dict(l=20, r=20, b=20, t=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.8)",
            ),
            height=600,
        )
        return fig

    def plot_similarity_heatmap(self, max_items: int = 35) -> go.Figure:
        """Create pairwise Cosine Similarity Matrix Heatmap between legal articles."""
        subset_n = min(max_items, self.n_samples)
        sub_vectors = self.vectors[:subset_n]
        sim_matrix = cosine_similarity(sub_vectors)

        labels = [f"#{i} {self.metadata[i].get('header', '')[:25]}" for i in range(subset_n)]

        fig = go.Figure(
            data=go.Heatmap(
                z=sim_matrix,
                x=labels,
                y=labels,
                colorscale="Viridis",
                hoverongaps=False,
                colorbar=dict(title="Cosine Sim"),
            )
        )

        fig.update_layout(
            title=dict(
                text=f"🔥 <b>Ma trận Tương đồng Cosine (Cosine Similarity Matrix)</b> - Top {subset_n} Điều luật",
                font=dict(size=16, color="#1E3A8A"),
            ),
            xaxis=dict(tickangle=45, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
            height=700,
            margin=dict(l=50, r=20, b=120, t=50),
        )
        return fig

    def plot_vector_weights_heatmap(self, vector_index: int) -> go.Figure:
        """Visualize raw 768-D float values of a specific vector as a 24x32 matrix heatmap."""
        if vector_index < 0 or vector_index >= self.n_samples:
            vector_index = 0

        vec = self.vectors[vector_index]
        meta = self.metadata[vector_index]

        # Reshape 768 into 24 rows x 32 columns
        grid_24_32 = vec.reshape((24, 32))

        fig = go.Figure(
            data=go.Heatmap(
                z=grid_24_32,
                colorscale="RdBu_r",
                zmid=0.0,
                colorbar=dict(title="Giá trị Float32"),
            )
        )

        header = meta.get("header", f"Vector #{vector_index}")
        fig.update_layout(
            title=dict(
                text=f"🧬 <b>Phổ nhiệt Trọng số 768 Chiều (Vector Embedding Heatmap)</b> - {header}",
                font=dict(size=15, color="#1E3A8A"),
            ),
            xaxis=dict(title="Cột Chiều (0..31)", gridcolor="#E2E8F0"),
            yaxis=dict(title="Hàng Chiều (0..23)", gridcolor="#E2E8F0"),
            height=420,
            margin=dict(l=30, r=20, b=30, t=50),
        )
        return fig
