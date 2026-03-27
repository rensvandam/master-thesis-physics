import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from matplotlib import style
import plotly.express as px

import project.model.coherence_from_data as coherence

def get_enabled_flags_for_detection_inefficiencies(INEFFICIENCIES, selected_options):
    """
    Returns a dictionary with the enabled flags for the selected inefficiencies.
    """
    NONE_OPTION = "None"
    ALL_OPTION = "All"
    if NONE_OPTION in selected_options:
        return {param: False for param in INEFFICIENCIES.values()}  # Disable all
    if ALL_OPTION in selected_options:
        return {param: True for param in INEFFICIENCIES.values()}  # Enable all
    
    # Enable only selected inefficiencies
    return {INEFFICIENCIES[opt]: True for opt in selected_options if opt in INEFFICIENCIES}


def show_coherence(x_data, y_data, show_fit=False, show_sigma=False, fit_method='with_k', auto_scale=False, title=None, save_as=None):
    """
    Plots the coherence using modern Streamlit standards. Optional: fit an exponential function to the coherence and plot that as well.
    
    Parameters
    ----------
    x_data : np.array()
        The amount of delay of the shifted signal.
    y_data : np.array()
        The auto-correlation of the photon count with itself.
    show_fit : bool
        If True, an exponential fit is calculated and plotted on top of the coherence. Default is False.
    show_sigma : bool
        If True, the standard deviation of the fitted value for n is shown in the legend. Default is False.
    auto_scale : bool
        If True, the y-axis is automatically scaled. Default is False.
    title : str
        The title that is put on top of the plot. Default is None.
    save_as : str
        The filepath for saving the plot. Default is None.
    """
    
    # Create a figure with improved styling
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set the style for a modern look
    style.use('ggplot')
    
    # Plot data with improved styling
    ax.plot(x_data, y_data, linewidth=2, color='#1f77b4', marker='o', markersize=3, alpha=0.7)
    
    if show_fit:
        if fit_method == 'with_k':
            fit, popt, pcov = coherence.fit_coherence_function(x_data, y_data, method='with_k', initial_guess=np.array([3, 2]))
            ax.plot(x_data, fit, color='#d62728', linewidth=2,
                    label=f"1 - (1/{np.round(popt[0], 2)})exp(-{np.round(popt[1], 2)}$\\tau$)")
            ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
            
        elif fit_method == 'without_k':
            fit, popt, pcov = coherence.fit_coherence_function(x_data, y_data, method = 'without_k', initial_guess=np.array([3]))#np.array([3, 2]))
            ax.plot(x_data, fit, color='#d62728', linewidth=2,
                    label=f"1 - (1/{np.round(popt[0], 2)})exp(-2.0092$\\tau$)")
            ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

        # Custom fit
        # fit, popt, pcov = coherence.fit_coherence_function(x_data, y_data, method='custom', initial_guess=np.array([3]))
        # ax.plot(x_data, fit, color='#d62728', linewidth=2,
        #         label=f"1 - (1/{np.round(popt[0], 2)})exp(-2.0092$\\tau$)")
        # ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    
    # Set labels with improved font styling
    ax.set_xlabel(r'$\tau \Delta t$ [ns]', fontsize=12)
    ax.set_ylabel(r'$g^{(2)}[\ell]$', fontsize=12)
    
    if show_sigma:
        # Add standard deviation of n as text to the plot
        sigma = np.sqrt(np.diag(pcov))[0]
        ax.text(0.5, 0.95, f"σ(n) = {np.round(sigma, 2)}", fontsize=10, ha='center', va='center', transform=ax.transAxes)

    # Handle y-axis limits
    if auto_scale:
        pass  # Let matplotlib handle the scaling automatically
    else:
        ax.set_ylim(-0.01, 1.25)
    
    # Apply tight layout
    plt.tight_layout()
    
    # Add title if provided
    if title:
        ax.set_title(title, fontsize=14, pad=10)
    
    # Save figure if path is provided
    if save_as:
        plt.savefig(fname=save_as, dpi=300, bbox_inches='tight')
    
    # Use Streamlit's modern method to display the plot
    st.pyplot(fig)
    
    # Add option to download the figure
    if save_as:
        with open(save_as, "rb") as file:
            btn = st.download_button(
                label="Download Figure",
                data=file,
                file_name=save_as.split('/')[-1],
                mime="image/png"
            )

# def show_coherence(x_data, y_data, show_fit=False, auto_scale=False, title=None, save_as=None):
#     """
#     Plots the coherence using Plotly for Streamlit with interactive features.
   
#     Parameters
#     ----------
#     x_data : np.array()
#         The amount of delay of the shifted signal.
#     y_data : np.array()
#         The auto-correlation of the photon count with itself.
#     show_fit : bool
#         If True, an exponential fit is calculated and plotted on top of the coherence. Default is False.
#     auto_scale : bool
#         If True, the y-axis is automatically scaled. Default is False.
#     title : str
#         The title that is put on top of the plot. Default is None.
#     save_as : str
#         The filepath for saving the plot. Default is None.
#     """
#     import plotly.graph_objects as go
#     import streamlit as st
#     import numpy as np
#     import io
#     from plotly.subplots import make_subplots
    
#     # Create a Plotly figure
#     fig = go.Figure()
    
#     # Add data points with markers
#     fig.add_trace(go.Scatter(
#         x=x_data,
#         y=y_data,
#         mode='lines+markers',
#         name='Coherence Data',
#         line=dict(color='#1f77b4', width=2),
#         marker=dict(size=6, opacity=0.7)
#     ))
    
#     # Add fit if requested
#     if show_fit:
#         # Using the coherence.fit_coherence_function from your original code
#         # You'll need to ensure this function is imported
#         fit, popt = coherence.fit_coherence_function(x_data, y_data, initial_guess=np.array([3, 2]))
        
#         fig.add_trace(go.Scatter(
#             x=x_data,
#             y=fit,
#             mode='lines',
#             name=f"1 - (1/{np.round(popt[0], 2)})exp(-{np.round(popt[1], 2)}τ)",
#             line=dict(color='#d62728', width=2)
#         ))
    
#     # Update layout with improved styling
#     fig.update_layout(
#         template="plotly_white",
#         title=title if title else None,
#         xaxis=dict(
#             title=r'$\tau \Delta t$ [ns]',
#             title_font=dict(size=14),
#             showgrid=True,
#             gridwidth=1,
#             gridcolor='rgba(211, 211, 211, 0.5)'
#         ),
#         yaxis=dict(
#             title=r'$g^{(2)}[\ell]$',
#             title_font=dict(size=14),
#             showgrid=True,
#             gridwidth=1,
#             gridcolor='rgba(211, 211, 211, 0.5)'
#         ),
#         legend=dict(
#             x=0.02,
#             y=0.98,
#             bgcolor='rgba(255, 255, 255, 0.8)',
#             bordercolor='rgba(0, 0, 0, 0.3)',
#             borderwidth=1
#         ),
#         margin=dict(l=60, r=20, t=40, b=60),
#         hoverlabel=dict(
#             bgcolor="white",
#             font_size=12,
#             font_family="Arial"
#         )
#     )
    
#     # Set y-axis limits if not auto scaling
#     if not auto_scale:
#         fig.update_yaxes(range=[-0.01, 1.25])
    
#     # Display the plot in Streamlit
#     st.plotly_chart(fig, use_container_width=True)
    
#     # Handle plot saving and download
#     if save_as:
#         # Save as static image first (needed for download button)
#         fig.write_image(save_as, scale=3)  # Higher scale for better resolution
        
#         # Create download button
#         with open(save_as, "rb") as file:
#             btn = st.download_button(
#                 label="Download Figure",
#                 data=file,
#                 file_name=save_as.split('/')[-1],
#                 mime="image/png"
#             )
        
#         # Also offer interactive HTML version
#         buffer = io.StringIO()
#         fig.write_html(buffer)
#         html_bytes = buffer.getvalue().encode()
        
#         st.download_button(
#             label="Download Interactive HTML",
#             data=html_bytes,
#             file_name=f"{save_as.split('.')[0]}.html",
#             mime="text/html"
#         )
    
#     return fig  # Return the figure object for further customization if needed


