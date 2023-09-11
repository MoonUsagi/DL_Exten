%RoadAndSkySegmentation Automation algorithm for pixel labeling.
%   RoadAndSkySegmentation is an automation algorithm for labeling pixels
%   in a scene using a semantic segmentation network trained on 11 classes
%   using the CamVid dataset. These classes include "Sky", "Building",
%   "Pole", "Road", "Pavement", "Tree", "SignSymbol", "Fence", "Car",
%   "Pedestrian" and "Bicyclist". This example only uses "Road" and "Sky".
%
%   See also groundTruthLabeler, imageLabeler, 
%   vision.labeler.AutomationAlgorithm.

% Copyright 2017 The MathWorks, Inc.

classdef RoadAndSkySegmentation < vision.labeler.AutomationAlgorithm
    
    %----------------------------------------------------------------------
    % Algorithm Description
    properties(Constant)

        %Name
        %   Character vector specifying name of algorithm.
        Name = 'RoadAndSkySegmentation'
        
        %Description
        %   Character vector specifying short description of algorithm.
        Description = 'This algorithm uses semanticseg with a pretrained network to annotate roads and sky'
        
        %UserDirections
        %   Cell array of character vectors specifying directions for
        %   algorithm users to follow in order to use algorithm.
        UserDirections = {...
            ['Automation algorithms are a way to automate manual labeling ' ...
            'tasks. This AutomationAlgorithm automatically creates pixel ', ...
            'level annotations for road and sky.'], ...
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
    
    %---------------------------------------------------------------------
    % Properties
    properties
        % Network saves the SeriesNetwork object that does the semantic
        % segmentation.
        PretrainedNetwork
        
        % Categories holds the default 'background', 'road', and 'sky'
        % categorical types.
        AllCategories = {'background'};
        
        % Store names for 'road' and 'sky'.
        RoadName
        SkyName
        
    end
    
    %----------------------------------------------------------------------
    % Setup
    methods
        
        function isValid = checkLabelDefinition(algObj, labelDef)
            
            % Allow any labels that are of type 'PixelLabel', and are named
            % 'Road' or 'Sky'.
            isValid = false;
            
            if (strcmpi(labelDef.Name, 'road') && labelDef.Type == labelType.PixelLabel)
                isValid = true;
                algObj.RoadName = labelDef.Name;
                algObj.AllCategories{end+1} = labelDef.Name;
            elseif (strcmpi(labelDef.Name, 'sky') && labelDef.Type == labelType.PixelLabel)
                isValid = true;
                algObj.SkyName = labelDef.Name;
                algObj.AllCategories{end+1} = labelDef.Name;
            elseif(labelDef.Type == labelType.PixelLabel)
                isValid = true;
            end
        end
    end
    
    %----------------------------------------------------------------------
    % Execution
    methods
        
        function initialize(algObj, ~)
            
            % Point to tempdir where pretrainedSegNet was downloaded.
            pretrainedFolder = fullfile(tempdir,'pretrainedSegNet');
            pretrainedSegNet = fullfile(pretrainedFolder,'segnetVGG16CamVid.mat'); 
            data = load(pretrainedSegNet);
            % Store the network in the 'Network' property of this object.
            algObj.PretrainedNetwork = data.net;
        end
        
        function autoLabels = run(algObj, I)
            
            % Setup categorical matrix with categories including road and
            % sky
            autoLabels = categorical(zeros(size(I,1), size(I,2)),0:2,algObj.AllCategories,'Ordinal',true);
            
            pixelCat = semanticseg(I, algObj.PretrainedNetwork);
            if ~isempty(pixelCat)
                % Add the selected label at the bounding box position(s)
                autoLabels(pixelCat == "Road") = algObj.RoadName;
                autoLabels(pixelCat == "Sky") = algObj.SkyName;
            end    
        end
    end
end