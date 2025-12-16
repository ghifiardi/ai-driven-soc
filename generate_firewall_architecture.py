import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_firewall_architecture_diagram():
    # Theme Colors
    BG_COLOR = '#1e1e1e'
    BOX_FACE = '#2d2d2d'
    BOX_EDGE = '#ffffff'
    TEXT_COLOR = '#ffffff'
    ARROW_COLOR = '#cccccc'
    GROUP_BG = '#333333'
    FW_COLOR = '#d93f0b' # Palo Alto Orange-ish

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Styles
    box_style = dict(boxstyle='round,pad=0.8', facecolor=BOX_FACE, edgecolor=BOX_EDGE, linewidth=1)
    
    def draw_box(x, y, text, width=3, height=1, style=box_style, fontsize=10, color=None):
        if color:
            style = style.copy()
            style['facecolor'] = color
        ax.text(x, y, text, ha='center', va='center', size=fontsize, color=TEXT_COLOR, bbox=style, zorder=10)
        return (x, y)

    def draw_arrow(start, end, label=None, rad=0.0):
        arrow_props = dict(arrowstyle='->', color=ARROW_COLOR, lw=1.5, connectionstyle=f"arc3,rad={rad}")
        ax.annotate("", xy=end, xytext=start, arrowprops=arrow_props, zorder=5)
        if label:
            mid_x = (start[0] + end[0]) / 2 + (0.5 if rad else 0)
            mid_y = (start[1] + end[1]) / 2 + (0.5 if rad else 0)
            if rad != 0: mid_y += rad
            ax.text(mid_x, mid_y, label, ha='center', va='center', color=TEXT_COLOR, fontsize=8, backgroundcolor=BG_COLOR, zorder=15)

    # --- Groups ---

    # MSSP Platform Group
    rect_plat = patches.Rectangle((0.5, 3.5), 6, 4, linewidth=1, edgecolor='#555555', facecolor=GROUP_BG, alpha=0.3, zorder=1)
    ax.add_patch(rect_plat)
    ax.text(3.5, 7.2, "MSSP Platform", ha='center', color=TEXT_COLOR, fontsize=12)

    # Customer/Tenant Group
    rect_cust = patches.Rectangle((8.5, 0.5), 5, 7, linewidth=1, edgecolor='#555555', facecolor=GROUP_BG, alpha=0.3, zorder=1)
    ax.add_patch(rect_cust)
    ax.text(11, 7.2, "Customer Network", ha='center', color=TEXT_COLOR, fontsize=12)

    # --- Nodes ---

    # Platform Components
    server = draw_box(3.5, 6, "Platform Server\n(Integration Logic)")
    db = draw_box(3.5, 4.5, "Multi-Tenant Config\n(Credentials)")
    agent = draw_box(1.5, 5.25, "AI Agent\n(TAA/MCP)", width=2)

    # Firewall Components
    ngfw = draw_box(11, 5.5, "Palo Alto NGFW\n(Direct API)", style=dict(boxstyle='round,pad=0.5', facecolor=FW_COLOR, edgecolor='#ffffff', linewidth=1))
    panorama = draw_box(11, 2.5, "Palo Alto Panorama\n(CMP)", style=dict(boxstyle='round,pad=0.5', facecolor=FW_COLOR, edgecolor='#ffffff', linewidth=1))
    
    # Managed Devices (under Panorama)
    fw_remote = draw_box(11, 1, "Managed FWs\n(Device Group)", width=2, fontsize=8)

    # --- Connections ---

    # Agent -> Server
    draw_arrow((1.5, 4.75), (3.5, 5.5), "Calls Tool\n(firewall_block_ip)", rad=-0.2)

    # Server -> DB
    draw_arrow((3.5, 5.5), (3.5, 5), "Reads Config")

    # Server -> NGFW (Integration)
    draw_arrow((4.8, 6), (9.8, 5.5), "XML API\n(Block IP)", rad=0)

    # Server -> Panorama (Integration)
    draw_arrow((4.8, 6), (9.8, 2.5), "XML API\n(Addr Group)", rad=-0.1)

    # Panorama -> Managed Devices
    draw_arrow((11, 2), (11, 1.5), "Push Config")

    # Title
    ax.text(7, 0.2, "Palo Alto Firewall Integration Architecture", ha='center', color=TEXT_COLOR, fontsize=14)

    plt.tight_layout()
    plt.savefig('firewall_integration_architecture.png', dpi=300, bbox_inches='tight', facecolor=BG_COLOR)
    print("Generated firewall_integration_architecture.png")

if __name__ == "__main__":
    create_firewall_architecture_diagram()
