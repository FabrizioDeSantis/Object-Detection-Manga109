def early_stopping(losses, patience):
    best_loss = float('inf')  # Inizializza il miglior valore di loss come infinito
    num_epochs_no_improvement = 0  # Contatore per tenere traccia del numero di epoche senza miglioramento
    
    for epoch, loss in enumerate(losses):
        if loss < best_loss:
            best_loss = loss
            num_epochs_no_improvement = 0
        else:
            num_epochs_no_improvement += 1

        if num_epochs_no_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch}!")
            return True
    return False

def check_annotation_validity(annotation, index, classes):
    for cl in classes[1:]:
        if len(annotation["page"][index][str(cl)])>0:
            continue
        else:
            return False
    return True

def load_all_images(parser, classes):
    """
    Using the parser from manga109api_custom, for every book load all the image paths and annotations and store them in a list of lists.
    Args:
      Parser (object)
    Returns:
      images (List[List[str, Dict]])
      author_labels (List[int])
    """
    images = []
    author_labels = []
    for book in parser.books:
      annotation, author = parser.get_annotation(book=book)
      for i in range(0, len(annotation["page"])):
        temp=[]
        if check_annotation_validity(annotation, i, classes):
          path=parser.img_path(book=book, index=i)
          temp.append(path)
          temp.append(annotation["page"][i])
          images.append(temp)
          author_labels.append(author)
    return images, author_labels