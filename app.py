import os
from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean

app = Flask(__name__)

# Load the data
# Update this to the location of your data file
df = pd.read_csv("nutritioncategory_fix.csv")

# Normalizing the data
scaler = MinMaxScaler()
features = ['energy_kcal', 'protein_g', 'fiber_total_g',
            'sugar_total_g', 'fa_sat_g', 'cholesterol_mg']
df_normalized = df.copy()
df_normalized[features] = scaler.fit_transform(df[features])


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        max_calories = float(request.form.get('max_calories'))
        protein_preference = request.form.get('protein_preference')
        food_category = request.form.get('food_category')

        # Filter based on user input
        filtered_df = df_normalized[df_normalized['energy_kcal']
                                    <= max_calories]
        if protein_preference == 'high':
            # 0.5 is just an example threshold
            filtered_df = filtered_df[filtered_df['protein_g'] >= 0.5]
        elif protein_preference == 'low':
            filtered_df = filtered_df[filtered_df['protein_g'] < 0.5]
        if food_category:
            filtered_df = filtered_df[filtered_df['category'] == food_category]
    else:
        filtered_df = df_normalized

    # Calculate similarity and make recommendations
    ideal_profile = [0 for _ in range(len(features))]
    filtered_df['similarity_score'] = filtered_df[features].apply(
        lambda x: euclidean(x, ideal_profile), axis=1)
    recommended_items = filtered_df.sort_values(
        by='similarity_score', ascending=True).head(10)

    return render_template("index.html", recommendations=recommended_items.to_dict('records'))


if __name__ == "__main__":
    app.run(debug=True,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))
