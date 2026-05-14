"""Prediction & Explanation tab component.

Renders an order input form on the left and a risk gauge + SHAP waterfall
on the right when the user clicks "Predict Risk".
"""

from datetime import datetime
from typing import Any

import streamlit as st

from adapters.ml.shap_explainer import ShapExplainer
from adapters.visualization.plotly_charts import risk_gauge, shap_waterfall
from application.use_cases import PredictSingleOrderUseCase
from domain.models import Order, OrderItem


def _build_order_from_form(
    shipping_mode: str,
    days: int,
    benefit: float,
    sales: float,
    profit: float,
) -> Order:
    """Construct a minimal Order from dashboard form inputs."""
    item = OrderItem(
        order_item_id=1,
        product_card_id="DASH-0001",
        quantity=1,
        unit_price=sales,
        discount=0.0,
        discount_rate=0.0,
        profit_ratio=profit / sales if sales > 0 else 0.0,
        sales=sales,
        item_total=sales,
    )
    return Order(
        order_id=0,
        order_date=datetime.now(),
        order_customer_id=0,
        order_region="North America",
        order_country="USA",
        order_state="CA",
        order_status="Complete",
        order_city="San Francisco",
        order_zipcode="00000",
        shipping_mode=shipping_mode,
        days_for_shipment_scheduled=days,
        benefit_per_order=benefit,
        sales_per_customer=sales,
        order_profit_per_order=profit,
        items=[item],
        late_delivery_risk=None,
    )


def render_prediction_tab(pipeline: dict[str, Any]) -> None:
    """Render the Predict & Explain tab.

    Left column: order input form.
    Right column: risk gauge + SHAP waterfall on button click.

    Args:
        pipeline: Dict of artifacts returned by load_pipeline().
    """
    st.header("Predict Late Delivery Risk for a Single Order")
    st.caption(
        "Fill in order details and click **Predict Risk** to get a real-time forecast."
    )

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("Order Details")

        shipping_mode = st.selectbox(
            "Shipping Mode",
            options=["First Class", "Second Class", "Standard Class", "Same Day"],
            index=2,
            help="First Class and Second Class have historically high late rates (95% and 77%).",
        )

        days = st.slider(
            "Scheduled Shipment Days",
            min_value=1,
            max_value=10,
            value=4,
            help="Number of days scheduled for shipment.",
        )

        sales = st.number_input(
            "Sales per Customer ($)",
            min_value=0.0,
            max_value=10000.0,
            value=200.0,
            step=10.0,
        )

        benefit = st.number_input(
            "Benefit per Order ($)",
            min_value=-500.0,
            max_value=2000.0,
            value=50.0,
            step=5.0,
        )

        profit = st.number_input(
            "Order Profit ($)",
            min_value=-1000.0,
            max_value=5000.0,
            value=30.0,
            step=5.0,
        )

        predict_clicked = st.button(
            "Predict Risk", type="primary", use_container_width=True
        )

    with right_col:
        if predict_clicked:
            order = _build_order_from_form(
                shipping_mode=shipping_mode,
                days=days,
                benefit=benefit,
                sales=sales,
                profit=profit,
            )

            encoder = pipeline["encoder"]
            model = pipeline["xgb_model"]
            feature_names = pipeline["feature_names"]
            explainer = ShapExplainer(model.model, feature_names)

            use_case = PredictSingleOrderUseCase(
                feature_encoder=encoder,
                model=model,
                explainer=explainer,
            )
            result = use_case.execute(order)

            # Risk gauge
            st.subheader("Risk Score")
            gauge_fig = risk_gauge(result.probability)
            st.plotly_chart(gauge_fig, use_container_width=True)

            # Risk label badge
            if result.risk_label:
                st.error(
                    f"LATE DELIVERY PREDICTED ({result.probability:.1%} probability)"
                )
            else:
                st.success(
                    f"ON-TIME DELIVERY PREDICTED ({result.probability:.1%} probability)"
                )

            # SHAP waterfall
            st.subheader("What Drove This Prediction?")
            local_result = explainer.explain_local(
                encoder.transform(
                    [
                        __import__(
                            "domain.services", fromlist=["extract_features"]
                        ).extract_features(order)
                    ]
                ).values,
                index=0,
            )
            waterfall_fig = shap_waterfall(
                local_result.shap_values, local_result.expected_value
            )
            st.plotly_chart(waterfall_fig, use_container_width=True)

            # Top 3 contributing factors
            sorted_shap = sorted(
                local_result.shap_values.items(),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )
            st.subheader("Top 3 Contributing Factors")
            for feature, value in sorted_shap[:3]:
                direction = "increases" if value > 0 else "reduces"
                st.markdown(
                    f"- **{feature}** — SHAP {value:+.4f} ({direction} late delivery risk)"
                )
        else:
            st.info("Fill in the order details on the left and click **Predict Risk**.")
            st.markdown(
                """
**How to interpret the gauge:**
- 🔴 > 70% → High Risk
- 🟡 40–70% → Medium Risk
- 🟢 < 40% → Low Risk

**SHAP values** show how each feature pushed the prediction up (red) or down (blue) from the base rate.
                """
            )
