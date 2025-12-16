import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_architecture_diagram():
    # Dark Theme Colors
    BG_COLOR = '#1e1e1e'
    BOX_FACE = '#2d2d2d'
    BOX_EDGE = '#ffffff'
    TEXT_COLOR = '#ffffff'
    ARROW_COLOR = '#cccccc'
    ARROW_COLOR = '#cccccc'
    GROUP_BG = '#333333'
    QUANTUM_COLOR = '#6929c4' # IBM Purple-ish
    
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Styles
    box_style = dict(boxstyle='round,pad=0.8', facecolor=BOX_FACE, edgecolor=BOX_EDGE, linewidth=1)
    diamond_style = dict(boxstyle='darrow,pad=0.5', facecolor=BOX_FACE, edgecolor=BOX_EDGE, linewidth=1) # Approximation for diamond
    # Matplotlib doesn't have a perfect diamond boxstyle with text inside easily, using a rotated square patch manually or just a box for now, 
    # but let's try to simulate a diamond shape for the decision node.
    
    def draw_box(x, y, text, width=2.5, height=1, style=box_style, fontsize=9):
        ax.text(x, y, text, ha='center', va='center', size=fontsize, color=TEXT_COLOR, bbox=style, zorder=10)
        return (x, y)

    def draw_arrow(start, end, label=None, rad=0.0):
        arrow_props = dict(arrowstyle='->', color=ARROW_COLOR, lw=1, connectionstyle=f"arc3,rad={rad}")
        ax.annotate("", xy=end, xytext=start, arrowprops=arrow_props, zorder=5)
        if label:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            # Adjust label position slightly based on curve
            if rad != 0:
                mid_y += rad * 2 
            ax.text(mid_x, mid_y, label, ha='center', va='center', color=TEXT_COLOR, fontsize=8, backgroundcolor=BG_COLOR, zorder=15)

    # --- Groups ---
    
    # Value Delivery Group (Left)
    rect_val = patches.Rectangle((0.5, 4), 4, 7.5, linewidth=1, edgecolor='#555555', facecolor=GROUP_BG, alpha=0.3, zorder=1)
    ax.add_patch(rect_val)
    ax.text(2.5, 11.2, "Value Delivery", ha='center', color=TEXT_COLOR, fontsize=10)

    # AI Core Group (Center)
    rect_core = patches.Rectangle((5, 2.5), 7, 9, linewidth=1, edgecolor='#555555', facecolor=GROUP_BG, alpha=0.3, zorder=1)
    ax.add_patch(rect_core)
    ax.text(8.5, 11.2, "AI Core (The Brain)", ha='center', color=TEXT_COLOR, fontsize=10)

    # --- Nodes ---

    # Left Column
    cla = draw_box(2.5, 10, "CLA: Learning Agent")
    portal = draw_box(2.5, 5, "Customer Portal")
    
    # New Left Nodes (Ecosystem)
    devsecops = draw_box(0.5, 8, "Proactive\nOrchestrator", width=2, style=dict(boxstyle='round,pad=0.5', facecolor='#4d4d4d', edgecolor='#00ff00', linewidth=1))
    phys_sec = draw_box(0.5, 2, "Physical Sec\n(YOLO)", width=2, style=dict(boxstyle='round,pad=0.5', facecolor='#4d4d4d', edgecolor='#ff9900', linewidth=1))
    
    # New Left Node (Fraud)
    fraud_dash = draw_box(0.5, 5, "Fraud\nDashboard", width=2, style=dict(boxstyle='round,pad=0.5', facecolor='#4d4d4d', edgecolor='#ff3333', linewidth=1))

    # New IBM Quantum Node (Differentiator)
    q_cloud = draw_box(8.5, 12, "IBM Quantum\nCloud", style=dict(boxstyle='round,pad=0.5', facecolor=QUANTUM_COLOR, edgecolor='#ffffff', linewidth=1))
    
    # Internal Quantum Agent
    q_agent = draw_box(6, 8, "Q-Agent\n(Threat)", width=2, style=dict(boxstyle='round,pad=0.5', facecolor='#2d2d2d', edgecolor=QUANTUM_COLOR, linewidth=1))

    # Center Column (AI Core)
    ada = draw_box(8.5, 10, "ADA: Anomaly Detection")
    taa = draw_box(8.5, 8, "TAA: Triage Analyst")
    mcp = draw_box(7, 6, "MCP Server")
    ext = draw_box(7, 4, "VirusTotal / Google SecOps")
    
    # Decision Diamond (Simulated)
    decision_x, decision_y = 10, 6
    diamond = patches.Polygon([[decision_x, decision_y+1], [decision_x+1.2, decision_y], [decision_x, decision_y-1], [decision_x-1.2, decision_y]], 
                              closed=True, facecolor=BOX_FACE, edgecolor=BOX_EDGE, linewidth=1, zorder=9)
    ax.add_patch(diamond)
    ax.text(decision_x, decision_y, "Action\nRequired?", ha='center', va='center', color=TEXT_COLOR, fontsize=8, zorder=10)
    decision = (decision_x, decision_y)

    # Right/Bottom Flow
    log_report = draw_box(13, 4, "Log & Report")
    cra = draw_box(14.5, 4, "CRA: Response Agent")
    
    # New Right Node (Digital Twin)
    digi_twin = draw_box(12, 10, "Digital Twin\nVisualization", width=2.5, style=dict(boxstyle='round,pad=0.5', facecolor='#003366', edgecolor='#00ccff', linewidth=1))
    
    # Vertical Stack on Right
    client_infra_top = draw_box(14.5, 2.5, "Client Infrastructure") # Target of Block
    
    client_infra_bottom = draw_box(14.5, 2.5, "Client Infrastructure") # Same node effectively
    ingest = draw_box(14.5, 1.2, "Unified Ingestion API")
    bq = draw_box(14.5, 0, "BigQuery Data Lake", style=dict(boxstyle='round,pad=0.8', facecolor=BOX_FACE, edgecolor=BOX_EDGE, linewidth=1)) 

    # --- Connections ---

    # CLA -> ADA (Model Updates)
    draw_arrow((2.5, 9.5), (7.2, 10), "Model Updates", rad=0.2)
    
    # ADA -> TAA (Alerts)
    draw_arrow((8.5, 9.5), (8.5, 8.5), "Alerts")
    
    # TAA -> Portal (Insights)
    draw_arrow((8.5, 7.5), (3.8, 5), "Insights") # Long arrow across
    
    # TAA -> MCP (Enrichment)
    draw_arrow((8.5, 7.5), (7, 6.5), "Enrichment")
    
    # MCP <-> Ext (Threat Intel)
    draw_arrow((7, 5.5), (7, 4.5), "Threat Intel") 
    
    # TAA -> Decision (Decision)
    draw_arrow((8.5, 7.5), (10, 6.8), "Decision")
    
    # Decision -> Log (No)
    draw_arrow((10.5, 5.5), (12, 4.5), "No", rad=-0.2)
    
    # Decision -> CRA (Yes)
    draw_arrow((10.5, 5.5), (14.5, 4.5), "Yes", rad=0.2)
    
    # CRA -> Client Infra (Block/Isolate)
    draw_arrow((14.5, 3.5), (14.5, 3), "Block/Isolate")
    
    # Client Infra -> Ingest (Logs/Events)
    draw_arrow((14.5, 2), (14.5, 1.7), "Logs/Events")
    
    # Ingest -> BQ (Routing)
    draw_arrow((14.5, 0.7), (14.5, 0.5), "Multi-Tenant Routing")
    
    # New Connections
    # DevSecOps -> Ingest (Vulnerability Reports)
    draw_arrow((1.5, 8), (13.2, 1.2), "Vuln Reports", rad=-0.5)
    
    # Phys Sec -> Ingest (Physical Alerts)
    draw_arrow((1.5, 2), (13.2, 1.2), "Physical Alerts")
    
    # Fraud -> Ingest (Fin. Alerts)
    draw_arrow((1.5, 5), (13.2, 1.2), "Fin. Alerts", rad=-0.3)
    
    # TAA -> Digital Twin (Real-time State)
    draw_arrow((9.8, 8), (10.8, 10), "State Sync")

    # TAA -> Q-Agent (Complex analysis)
    draw_arrow((8.5, 8), (7, 8), "Deep\nAnalysis", rad=0)

    # Q-Agent -> IBM Quantum (Kernel processing)
    draw_arrow((6, 8.5), (8.5, 11.5), "Q-Kernel\nCompute", rad=0.2)

    plt.tight_layout()
    plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight', facecolor=BG_COLOR)
    print("Generated architecture_diagram.png")

if __name__ == "__main__":
    create_architecture_diagram()
