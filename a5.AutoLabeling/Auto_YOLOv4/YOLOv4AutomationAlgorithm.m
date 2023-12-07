classdef YOLOv4AutomationAlgorithm < vision.labeler.AutomationAlgorithm
    
    %   Copyright 2017-2022 The MathWorks, Inc.
    properties(Constant)
        
        % Name: Character vector specifying the name of the algoritm.
        Name = 'YOLOv4 Object Detection Automation';
        
        % Description: One-line description of the algorithm.
        Description = ['This example uses pre-trained YOLOv4 ' ...
            'detector to perform object labelling.'];
        
        UserDirections = {...
            ['This AutomationAlgorithm automatically creates bounding box ', ...
           'labels for 12 object categories.'], ...
           ['Review and Modify: Review automated labels over the interval ', ...
           'using playback controls. Modify/delete/add ROIs that were not ' ...
           'satisfactorily automated at this stage. If the results are ' ...
           'satisfactory, click Accept to accept the automated labels.'], ...
           ['Accept/Cancel: If results of automation are satisfactory, ' ...
           'click Accept to accept all automated labels and return to ' ...
           'manual labeling. If results of automation are not ' ...
           'satisfactory, click Cancel to return to manual labeling ' ...
           'without saving automated labels.']};

    end
    
    properties
        
        Model

        % Threshold for the object detection score
        Threshold = 0.3
        
        % Label class names (super-classes)
        Labels = {'person','vehicle','outdoor','animal','accessory','sports', ...
        'kitchen','food','furniture','electonic','appliance','indoor' };

        % IDs corresponding to the labels. Note that be group together
        % similar classes into superclasses defined below:

        % ["person"]  = person
        % ["bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat"]  = vehicle
        % ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench"] = outdoor
        % ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"] = animal
        % ["backpack", "umbrella", "handbag", "tie", "suitcase"]; = accessory
        % ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket"] = sports
        % ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"] = kitchen
        % ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"] = food
        % ["chair", "sofa", "pottedplant", "bed", "diningtable", "toilet"] = furniture
        % ["tvmonitor", "laptop", "mouse", "remote" , "keyboard", "cell phone"] = electronic
        % ["microwave", "oven", "toaster", "sink", "refrigerator"] =appliance
        % ["book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"] = indoor

        LabelIDs = {1, 2:9, 10:14, 15:24, 25:29, 30:39, 40:46, 47:56, 57:62, 63:68, 69:73, 74:80};
        
        % Dictionary containing LabelID to Label mapping.
        Map = dictionary;

        
    end
    
    
    methods
       
        function isValid = checkLabelDefinition(algObj, labelDef)
                     
            isValid = false;

            % We turn on only those labels whose name matches the list
            if any(strcmp(algObj.Labels,labelDef.Name))
                isValid = true;
            end
                        
        end
        
        function isReady = checkSetup(algObj)
 
            isReady = 0;
            if ~isempty(algObj.ValidLabelDefinitions)
                isReady = 1;
            end


        end
        
        
    end
    

    methods

        function initialize(algObj, ~)
            
           % Load the detector.
           algObj.Model = yolov4ObjectDetector("csp-darknet53-coco"); 
           
           % Populate the dictionary for mapping label IDs with label names.
           for i=1:80
               idx = find(cellfun(@(x) ismember(i,x),algObj.LabelIDs));
               algObj.Map(i) = algObj.Labels(idx);
           end
            
        end
        

        function autoLabels = run(algObj, I)

            % Perform detection.
            [bboxes, scores, labels] = detect(algObj.Model,I , Threshold=algObj.Threshold);            

            autoLabels = struct('Name', cell(1, size(bboxes, 1) ), ...
                'Type', cell(1, size(bboxes, 1) ),'Position',zeros([1 4]));

            for i=1:size(bboxes, 1)               
                % Add the predicted label to outputs
                currentLabel = algObj.Map(double(labels(i)));
                autoLabels(i).Name     = currentLabel{:};
                autoLabels(i).Type     = labelType.Rectangle;
                autoLabels(i).Position = bboxes(i,:);
        
            end
            
            
        end
        

    end
end
