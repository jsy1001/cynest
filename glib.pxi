cdef extern from "glib.h":

    ctypedef void * gpointer

    ctypedef struct GList

    ctypedef struct GList:
        gpointer data
        GList *next
        GList *prev

    GList *g_list_append(GList *, gpointer)
    void g_list_free(GList *)
    int g_list_length(GList *)

    ctypedef struct GHashTable

    gpointer g_hash_table_lookup(GHashTable *, gpointer)


# Local Variables:
# mode: python
# End:
