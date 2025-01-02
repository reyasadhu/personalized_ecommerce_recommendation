# personalized_ecommerce_recommendation
A retrieval based recommendation system with two tower architecture

## Key Features
- Two-tower architecture for scalable recommendations
- Multi-modal feature processing (textual, categorical, numerical, temporal)
- Efficient embedding layers for sparse features
- User interaction history integration
- Representing user and item embedding in same vector space for approximate KNN search

## Implementation Details
- Built using TensorFlow/Keras
- Supports batch processing
- Handles variable-length input sequences
- Optimized for large-scale recommendation tasks

## Installation

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

## Two Tower Architecture

### Item Tower
- Processes item features through embedding layers
- Handles features like:
  - Property_1 through Property_11: These are item metadata, where the texts are hashed. Some of the properties are numeric, where the rest are used by textvectorization.
  - Category IDs: Category of an item
  - Parent level hierarchies: Hierarchical categories are embedded to capture item relationships

### User Tower
- Processes user interaction history
- Handles user features and behavioral patterns
- Creates user embeddings based on historical interactions
- Time-based features are processed to capture temporal patterns

## Dataset

The raw data can be downloaded from here https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset/data
Save the data in a folder raw_data to run the code.

## Baseline Model
This project implements and evaluates several baseline recommendation approaches:
- Most Popular Items: Suggests the most frequently interacted-with items, ranked by their overall popularity.
- Recently Viewed Items: Recommends items a user has recently viewed, ranked based on how recently they were interacted with.
- User-User Collaborative Filtering: Identifies 10 similar users using cosine similarity and recommends items they have interacted with, ranked by interaction frequency or type.
- Item-Item Collaborative Filtering: Finds items similar to those the user has already interacted with and ranks them based on similarity.
- Singular Value Decomposition (SVD): A matrix factorization technique used to predict missing values in the interaction matrix.

### Key Findings
1. Popular items baseline performs poorly, indicating personalization is important.
2. Recently Viewed Items performs best for immediate recommendations (P@1)
3. Item-Item CF shows best overall recall.
5. Event weighting (view=1, cart=2, purchase=3) improves collaborative filtering performance

### Implementation Details
- Uses sparse matrices for efficient computation
- Implements cosine similarity for user/item similarity
- Evaluates with Recall@K metric
