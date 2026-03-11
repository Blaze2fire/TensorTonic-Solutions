def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """

    x1a,y1a,x2a,y2a=box_a
    x1b,y1b,x2b,y2b=box_b

    inter_x1=max(x1a,x1b);inter_x2=min(x2a,x2b)
    inter_y1=max(y1a,y1b);inter_y2=min(y2a,y2b)
    AreaA=((x2a-x1a)*(y2a-y1a))

    inter_width=max(0,inter_x2-inter_x1)
    inter_height=max(0,inter_y2-inter_y1)
    AreaB=(x2b-x1b)*(y2b-y1b)

    IArea=abs(inter_width*inter_height)
    UArea=AreaA+AreaB-IArea
    

    return (IArea/UArea if UArea>0 else 0)
    pass