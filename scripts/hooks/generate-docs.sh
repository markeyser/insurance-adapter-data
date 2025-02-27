# #!/bin/bash

# # Generate the project documentation
# pydoc-markdown -I src \
#   -m fpservicesagemodel.app \
#   -m fpservicesagemodel.main \
#   -m fpservicesagemodel.data_preprocessing.data_chunking.data_chunking \
#   -m fpservicesagemodel.data_preprocessing.data_chunking.data_chunking_utils \
#   -m fpservicesagemodel.data_preprocessing.data_extraction.data_cleaning \
#   -m fpservicesagemodel.data_preprocessing.data_extraction.data_extraction_utils \
#   -m fpservicesagemodel.data_preprocessing.data_extraction.data_reader \
#   -m fpservicesagemodel.embeddings.embedding_generator \
#   -m fpservicesagemodel.embeddings.embedding_utils \
#   -m fpservicesagemodel.evaluation.evaluation_metrics \
#   -m fpservicesagemodel.evaluation.evaluation_utils \
#   -m fpservicesagemodel.retrievers.retriever \
#   -m fpservicesagemodel.retrievers.retriever_preprocessing \
#   -m fpservicesagemodel.retrievers.retriever_utils \
#   -m fpservicesagemodel.generators.generation_postprocessing \
#   -m fpservicesagemodel.generators.generator \
#   -m fpservicesagemodel.generators.generator_utils \
#   --render-toc > docs/api-reference.md

# # Automatically stage the generated documentation
# git add docs/api-reference.md

# # Always exit with success
# exit 0
