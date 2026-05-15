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
from domain.services import extract_features


def _build_order_from_form(
    shipping_mode: str,
    days: int,
    region: str,
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
        order_region=region,
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
    """Render the Predict & Explain tab."""
    st.header("Predict Late Delivery Risk")
    st.warning(
        "**Demo Mode** — This predictor uses a model trained on a 1,000-order "
        "sample for real-time interactivity. Predictions illustrate how the model "
        "works, not production-grade accuracy. "
        "See the **Model Results** tab (Full Dataset) for metrics from 180K orders."
    )

    left_col, spacer, right_col = st.columns([4, 1, 6])

    with left_col:
        st.markdown("##### Order Details")

        shipping_mode = st.selectbox(
            "Shipping Mode",
            options=["Standard Class", "First Class", "Second Class", "Same Day"],
            index=0,
            help="First Class (95% late) and Second Class (77% late) are high-risk.",
        )

        days = st.slider(
            "Scheduled Shipment Days",
            min_value=1,
            max_value=10,
            value=4,
        )

        region = st.selectbox(
            "Order Region",
            options=[
                "Central America",
                "Eastern Europe",
                "North America",
                "Oceania",
                "South America",
                "Southeast Asia",
                "Southern Africa",
                "Western Europe",
            ],
            index=2,
        )

        c1, c2 = st.columns(2)
        with c1:
            sales = st.number_input(
                "Sales ($)", min_value=0.0, max_value=10000.0, value=200.0, step=10.0
            )
            benefit = st.number_input(
                "Benefit ($)", min_value=-500.0, max_value=2000.0, value=50.0, step=5.0
            )
        with c2:
            profit = st.number_input(
                "Profit ($)", min_value=-1000.0, max_value=5000.0, value=30.0, step=5.0
            )

        st.markdown("")
        predict_clicked = st.button(
            "🔮  Predict Risk", type="primary", use_container_width=True
        )

    with right_col:
        if predict_clicked:
            order = _build_order_from_form(
                shipping_mode=shipping_mode,
                days=days,
                region=region,
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
            gauge_fig = risk_gauge(result.probability)
            st.plotly_chart(gauge_fig, use_container_width=True)

            # Risk label badge
            if result.risk_label:
                st.error(
                    f"**LATE DELIVERY PREDICTED** — {result.probability:.1%} probability"
                )
            else:
                st.success(
                    f"**ON-TIME DELIVERY EXPECTED** — {result.probability:.1%} probability"
                )

            # SHAP waterfall
            st.markdown("##### What Drove This Prediction?")
            local_result = explainer.explain_local(
                encoder.transform([extract_features(order)]).values,
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
            st.markdown("##### Top Contributing Factors")
            for feature, value in sorted_shap[:3]:
                direction = "increases" if value > 0 else "reduces"
                icon = "🔺" if value > 0 else "🔽"
                st.markdown(
                    f"{icon} **{feature}** — SHAP {value:+.4f} ({direction} risk)"
                )
        else:
            st.markdown("")
            st.markdown("")
            st.info(
                "👈 Fill in order details and click **Predict Risk** to get a "
                "real-time risk score with SHAP explanation."
            )
            st.markdown("")

            with st.expander("How to read the results", expanded=True):
                st.markdown(
                    """
**Risk Gauge:**
- 🔴 **> 70%** — High Risk → reroute or notify customer
- 🟡 **40–70%** — Medium Risk → monitor closely
- 🟢 **< 40%** — Low Risk → proceed normally

**SHAP Waterfall:**
Each bar shows how a feature pushed the prediction UP (toward late)
or DOWN (toward on-time) from the base rate.

**Try this:** Change Shipping Mode from Standard Class to First Class
and watch the risk gauge jump — shipping mode is the dominant signal.
                    """
                )
