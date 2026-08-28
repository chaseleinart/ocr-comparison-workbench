import asyncio
import tempfile
import os

from PIL import Image
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import streamlit as st

from ocr_evaluation import evaluate_invoice_ocr
from model_service import get_all_ocr_model_names, run_ocr_inference

load_dotenv()

# Set page config
st.set_page_config(page_title="Open-source OCR Model Playground", layout="wide")

# Custom CSS for responsive code containers
st.markdown(
    """
<style>
    .stMarkdown {
        width: 100%;
    }
    pre {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        max-width: 100% !important;
    }
    code {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        max-width: 100% !important;
    }
    .streamlit-expanderContent {
        width: 100% !important;
    }
    div[data-testid="stCodeBlock"] {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        max-width: 100% !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "image_url" not in st.session_state:
    st.session_state.image_url = None
if "reference_text" not in st.session_state:
    st.session_state.reference_text = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "last_generated_text" not in st.session_state:
    st.session_state.last_generated_text = None
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# Get all available model names
all_models = get_all_ocr_model_names()

# Validate that we have models available
if not all_models:
    st.error("No models are available. Please check your configuration.")
    st.stop()

# Ensure default model is valid
default_model = st.session_state.selected_model
if default_model not in all_models:
    default_model = all_models[0] if all_models else "DeepSeek OCR"

# Update session state if default changed
if default_model != st.session_state.selected_model:
    st.session_state.selected_model = default_model


# Define OCR request handler function
async def handle_ocr_request():
    # Check if dots OCR model is selected
    is_dots_ocr = st.session_state.selected_model == "Dots OCR"

    # Validate input based on model type
    if is_dots_ocr:
        if not st.session_state.image_url:
            st.error("Please enter an image URL for Dots OCR!")
            return
        image_input = st.session_state.image_url.strip()
    else:
        if not st.session_state.uploaded_image:
            st.error("Please upload an image first!")
            return
        image_input = None

    # Set processing state
    st.session_state.is_processing = True

    # Save uploaded image to file (only for non-dots OCR models)
    image_file = None
    try:
        if not is_dots_ocr:
            # For other models, use temporary file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f".{st.session_state.uploaded_image.name.split('.')[-1]}",
            ) as tmp_file:
                # Write the uploaded image to the temporary file
                tmp_file.write(st.session_state.uploaded_image.read())
                image_file = tmp_file.name
            image_input = image_file

            # Reset file pointer for potential future use
            st.session_state.uploaded_image.seek(0)

        # Run OCR inference with image path or URL
        with st.spinner("Processing OCR... Please wait."):
            response_text = await run_ocr_inference(
                ocr_model_name=st.session_state.selected_model, image_input=image_input
            )

        st.session_state.last_generated_text = response_text.strip()

    except Exception as e:
        st.session_state.last_generated_text = f"Error processing OCR: {str(e)}"
    finally:
        # Clean up temporary file (only for non-dots OCR models)
        if not is_dots_ocr and image_file and os.path.exists(image_file):
            os.unlink(image_file)

        # Clear processing state
        st.session_state.is_processing = False


# Sidebar configuration
with st.sidebar:
    st.title("Configuration")

    # Model selection section
    st.write("### Select Model")
    model = st.selectbox(
        "Select Model",
        options=all_models,
        index=(
            all_models.index(st.session_state.selected_model)
            if st.session_state.selected_model in all_models
            else 0
        ),
        key="model_select",
    )

    # Update session state when model changes
    if model != st.session_state.selected_model:
        st.session_state.selected_model = model
        # Clear previous results when model changes
        st.session_state.last_generated_text = None
        st.session_state.evaluation_results = None
        st.session_state.uploaded_image = None
        st.session_state.image_url = None

    # Check if dots OCR model is selected
    is_dots_ocr = st.session_state.selected_model == "Dots OCR"

    # Show appropriate input based on model selection
    if is_dots_ocr:
        # For Dots OCR, show URL input
        image_url_input = st.text_input(
            "Image URL for OCR",
            value=st.session_state.image_url or "",
            help="Enter a URL to an image containing text to extract",
            placeholder="https://example.com/image.png",
        )
        st.session_state.image_url = image_url_input if image_url_input else None
        st.session_state.uploaded_image = None  # Clear uploaded image
    else:
        # For other models, show file uploader
        uploaded_file = st.file_uploader(
            "Upload Image for OCR",
            type=["png", "jpg", "jpeg"],
            help="Upload an image containing text to extract",
        )

        if uploaded_file is not None:
            st.session_state.uploaded_image = uploaded_file
        else:
            st.session_state.uploaded_image = None
        st.session_state.image_url = None  # Clear URL

    # Extract Text button
    if st.button("Extract Text 🔍", type="primary"):
        if is_dots_ocr:
            if not st.session_state.image_url:
                st.error("Please enter an image URL first!")
            else:
                try:
                    # Validate that selected model is still available
                    all_models_check = get_all_ocr_model_names()
                    if st.session_state.selected_model not in all_models_check:
                        st.error(
                            "Selected model is no longer available. Please reselect a model."
                        )
                    else:
                        asyncio.run(handle_ocr_request())
                        st.rerun()
                except Exception as e:
                    st.error(f"An error occurred during OCR: {str(e)}")
                    st.error("Please try again or check your configuration.")
                    st.session_state.is_processing = False
        else:
            if not st.session_state.uploaded_image:
                st.error("Please upload an image first!")
            else:
                try:
                    # Validate that selected model is still available
                    all_models_check = get_all_ocr_model_names()
                    if st.session_state.selected_model not in all_models_check:
                        st.error(
                            "Selected model is no longer available. Please reselect a model."
                        )
                    else:
                        asyncio.run(handle_ocr_request())
                        st.rerun()
                except Exception as e:
                    st.error(f"An error occurred during OCR: {str(e)}")
                    st.error("Please try again or check your configuration.")
                    st.session_state.is_processing = False

    # Image Preview in sidebar
    st.write("### Image Preview")
    has_image_input = st.session_state.uploaded_image is not None or (
        st.session_state.selected_model == "Dots OCR" and st.session_state.image_url
    )

    if has_image_input:
        if st.session_state.selected_model == "Dots OCR" and st.session_state.image_url:
            # For Dots OCR with URL, try to display the image from URL
            try:
                st.image(
                    st.session_state.image_url,
                    caption="Image from URL",
                    use_column_width=True,
                )
            except Exception as e:
                st.warning(f"Could not display image from URL: {str(e)}")
                st.info("Image URL provided, but preview unavailable")
        elif st.session_state.uploaded_image:
            image = Image.open(st.session_state.uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        st.info("Upload an image to see a preview")

    st.session_state.reference_text = st.text_area(
        "Reference Text",
        help="Enter reference/ground truth text to compare against",
        height=200,
    )
    # if not st.session_state.reference_text:
    #     st.error("Please enter a reference text to evaluate the OCR result.")
    #     st.stop()

    # Evaluation section
    st.write("### Evaluation")
    if st.button("Evaluate OCR Result"):
        if not st.session_state.last_generated_text:
            st.error("Please generate OCR text first")
        elif not st.session_state.reference_text:
            st.error("Please enter reference text before evaluating")
        else:
            try:
                with st.spinner("Evaluating OCR result..."):
                    st.session_state.evaluation_results = evaluate_invoice_ocr(
                        st.session_state.last_generated_text,
                        st.session_state.reference_text,
                    )
                st.success("Evaluation complete!")
            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")
                st.error("Please try again or check your evaluation configuration.")

# Main interface
col_title, col_clear = st.columns([3, 1])
with col_title:
    st.title("Open-source OCR Model Playground")
    powered_by_html = """
        <div style='display: flex; align-items: center; gap: 10px; margin-top: -10px;'>
            <span style='font-size: 20px; color: #666;'>Featuring</span>
            <img src="https://github.com/sitammeur/test-assets/blob/main/datalab-logo.png?raw=true" width="40">
            <img src="https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/dark/deepseek-color.png" width="60">
            <img src="https://cdn-avatars.huggingface.co/v1/production/uploads/620760a26e3b7210c2ff1943/-s1gyJfvbE1RgO5iBeNOi.png" width="45">
            <img src="https://cdn-avatars.huggingface.co/v1/production/uploads/63c64dd877caf00391004e20/aWC70TyF2UhxyaUh1alpu.png" width="45">
        </div>
    """
    st.markdown(powered_by_html, unsafe_allow_html=True)

with col_clear:
    st.write("")  # Spacing
    st.write("")  # Spacing
    if st.button("Clear 🗑️", type="secondary"):
        st.session_state.uploaded_image = None
        st.session_state.image_url = None
        st.session_state.last_generated_text = None
        st.session_state.evaluation_results = None
        st.session_state.is_processing = False
        st.rerun()

# Main content area - extracted text display
st.write("### Extracted Text")
st.write(f"**Model:** {st.session_state.selected_model}")

# Show final extracted text
if st.session_state.last_generated_text:
    st.markdown(st.session_state.last_generated_text, unsafe_allow_html=True)
else:
    has_image_input = st.session_state.uploaded_image is not None or (
        st.session_state.selected_model == "Dots OCR" and st.session_state.image_url
    )
    if has_image_input:
        st.info(
            "Click 'Extract Text 🔍' button in the sidebar to extract text from the image."
        )
    else:
        if st.session_state.selected_model == "Dots OCR":
            st.info("Please enter an image URL in the sidebar to begin OCR extraction.")
        else:
            st.info("Please upload an image from the sidebar to begin OCR extraction.")

# Display evaluation results
if st.session_state.evaluation_results:
    try:
        st.write("---")
        st.header("Evaluation results generated using DeepEval")

        # Validate evaluation results structure
        def validate_evaluation_result(result):
            if not result or not isinstance(result, dict):
                return False
            if "detailed_metrics" not in result or "overall_score" not in result:
                return False
            required_metrics = [
                "field_accuracy",
                "line_item_quality",
                "financial_consistency",
            ]
            for metric in required_metrics:
                if metric not in result["detailed_metrics"]:
                    return False
                if "score" not in result["detailed_metrics"][metric]:
                    return False
            return True

        if not validate_evaluation_result(st.session_state.evaluation_results):
            st.error("Invalid evaluation result structure")
        else:
            # Create single model plot
            plot_data = pd.DataFrame(
                {
                    "Metric": [
                        "Field Accuracy",
                        "Line Item Quality",
                        "Financial Consistency",
                        "Overall Score",
                    ],
                    "Score": [
                        st.session_state.evaluation_results["detailed_metrics"][
                            "field_accuracy"
                        ]["score"],
                        st.session_state.evaluation_results["detailed_metrics"][
                            "line_item_quality"
                        ]["score"],
                        st.session_state.evaluation_results["detailed_metrics"][
                            "financial_consistency"
                        ]["score"],
                        st.session_state.evaluation_results["overall_score"],
                    ],
                }
            )

            fig = px.bar(
                plot_data,
                x="Metric",
                y="Score",
                title=f"{st.session_state.selected_model} Performance Metrics",
                template="plotly_dark",
                color="Score",
                color_continuous_scale="Viridis",
            )

            fig.update_layout(
                xaxis_title="Evaluation Metrics",
                yaxis_title="Score",
                plot_bgcolor="rgba(32, 32, 32, 1)",
                paper_bgcolor="rgba(32, 32, 32, 1)",
                bargap=0.2,
                font=dict(color="#E0E0E0"),
                title_font=dict(color="#E0E0E0"),
                showlegend=False,
            )

            fig.update_xaxes(
                gridcolor="rgba(128, 128, 128, 0.2)",
                zerolinecolor="rgba(128, 128, 128, 0.2)",
            )
            fig.update_yaxes(
                gridcolor="rgba(128, 128, 128, 0.2)",
                zerolinecolor="rgba(128, 128, 128, 0.2)",
            )

            st.plotly_chart(fig, use_container_width=True)

            st.write(f"### {st.session_state.selected_model} detailed metrics")

            metrics_data = []
            for metric in [
                "field_accuracy",
                "line_item_quality",
                "financial_consistency",
            ]:
                metric_display_name = {
                    "field_accuracy": "Field Accuracy",
                    "line_item_quality": "Line Item Quality",
                    "financial_consistency": "Financial Consistency",
                }.get(metric, metric.title())
                row = {
                    "Metric": metric_display_name,
                    "Score": f"{st.session_state.evaluation_results['detailed_metrics'][metric]['score']:.2f}",
                    "Reasoning": st.session_state.evaluation_results[
                        "detailed_metrics"
                    ][metric]["reason"],
                }
                metrics_data.append(row)

            metrics_data.append(
                {
                    "Metric": "Overall Score",
                    "Score": f"{st.session_state.evaluation_results['overall_score']:.2f}",
                    "Reasoning": "Final weighted average",
                }
            )

            # Display metrics table
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(
                metrics_df,
                column_config={
                    "Metric": st.column_config.TextColumn("Metric", width="small"),
                    "Score": st.column_config.TextColumn("Score", width="small"),
                    "Reasoning": st.column_config.TextColumn(
                        "Reasoning", width="large"
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )
    except Exception as e:
        st.error(f"Error displaying evaluation results: {str(e)}")
        st.error("Please try running the evaluation again.")