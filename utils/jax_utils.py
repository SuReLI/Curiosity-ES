import jax
import jax.numpy as jnp

@jax.jit
def knn_jax_single(target, batch):
    """    
    Args:
        target (jax.numpy.ndarray): Vecteur cible de dimension (1, d).
        batch (jax.numpy.ndarray): Batch de vecteurs, de dimension (batch_size, d).

    Returns:
        jax.numpy.ndarray: Distances L2 entre target et chaque vecteur du batch.
    """
    return jnp.linalg.norm(batch - target, axis=1)



# def knn_jax(x, y, k):
#     """    
#     Args:
#         x (jax.numpy.ndarray): Vecteur cible de dimension (1, d).
#         y (jax.numpy.ndarray): Batch de vecteurs, de dimension (batch_size, d).
#         k (int): Nombre de plus proches voisins à considérer.

#     Returns:
#         float: Somme des distances des k plus proches voisins dans y pour le vecteur x.
#     """
#     # Calcul des distances L2 entre x et chaque vecteur du batch y
#     distances = knn_jax_single(x, y)

#     # Tri des indices des distances dans l'ordre croissant
#     sorted_indices = jnp.argsort(distances)

#     # Sélection des k indices des plus proches voisins
#     k_nearest_indices = sorted_indices[:k]

#     # Calcul de la somme des distances des k plus proches voisins
#     k_nearest_distances = distances[k_nearest_indices]
#     k_nearest_distances_sum = jnp.sum(k_nearest_distances)

#     return k_nearest_distances_sum

# def knn_jax_parallel(x_array, y_array, k):
#     """    
#     Args:
#         x_array (jax.numpy.ndarray): Tableau des vecteurs cibles, de dimension (num_pairs, 1, d).
#         y_array (jax.numpy.ndarray): Tableau des batchs de vecteurs, de dimension (num_pairs, batch_size, d).
#         k (int): Nombre de plus proches voisins à considérer.

#     Returns:
#         jax.numpy.ndarray: Tableau des sommes des distances des k plus proches voisins pour chaque paire de x et y.
#     """
#     # Appel en parallèle de la fonction knn_jax pour chaque paire de x et y
#     parallel_knn_jax = jax.vmap(knn_jax, in_axes=(0, 0, None))
#     results = parallel_knn_jax(x_array, y_array, k)

#     return results

# if __name__=='__main__':

