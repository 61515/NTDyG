def sort_event_frame(frame, order_column, source_row_column="__source_row"):
    """Sort events by the stored order key, breaking ties by original CSV row."""
    order_columns = [order_column]
    if source_row_column in frame.columns:
        order_columns.append(source_row_column)
    return frame.sort_values(order_columns, kind="stable").reset_index(drop=True)


def select_normal_events(frame, label_column):
    """Return only normal-labeled events for the memory model's ranking loss."""
    return frame.loc[frame[label_column] == 0].copy().reset_index(drop=True)
