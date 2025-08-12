"use client"

import { useState, useRef } from "react"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Upload, X, Sparkles, Download, RotateCcw } from "lucide-react"

export default function VirtualTryOn() {
  const [personImage, setPersonImage] = useState(null)
  const [clothingImage, setClothingImage] = useState(null)
  const [resultImage, setResultImage] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [dragOver, setDragOver] = useState({ person: false, clothing: false })

  const personInputRef = useRef(null)
  const clothingInputRef = useRef(null)

  const handleImageUpload = (file, type) => {
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader()
      reader.onload = (e) => {
        if (type === "person") {
          setPersonImage(e.target.result)
        } else {
          setClothingImage(e.target.result)
        }
      }
      reader.readAsDataURL(file)
    }
  }

  const handleDrop = (e, type) => {
    e.preventDefault()
    setDragOver({ ...dragOver, [type]: false })
    const files = e.dataTransfer.files
    if (files.length > 0) {
      handleImageUpload(files[0], type)
    }
  }

  const handleDragOver = (e, type) => {
    e.preventDefault()
    setDragOver({ ...dragOver, [type]: true })
  }

  const handleDragLeave = (e, type) => {
    e.preventDefault()
    setDragOver({ ...dragOver, [type]: false })
  }

  const removeImage = (type) => {
    if (type === "person") {
      setPersonImage(null)
    } else {
      setClothingImage(null)
    }
  }

  const processVirtualTryOn = async () => {
    if (!personImage || !clothingImage) return

    setIsProcessing(true)

    // Simulate API call to VITON model
    setTimeout(() => {
      // For demo purposes, we'll use a placeholder result
      setResultImage("/home/akansh_26/Downloads/00891_01430_00 (4).png")
      setIsProcessing(false)
    }, 3000)
  }

  const downloadResult = () => {
    if (resultImage) {
      const link = document.createElement("a")
      link.href = resultImage
      link.download = "virtual-try-on-result.png"
      link.click()
    }
  }

  const resetAll = () => {
    setPersonImage(null)
    setClothingImage(null)
    setResultImage(null)
  }

  return (
    <div className="min-h-screen bg-amber-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <h1 className="text-4xl font-bold text-[#a00a11]">Virtual Try-On</h1>
          </div>
          <p className="text-gray-600 text-lg">Upload your photo and clothing item to see how it looks on you!</p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Person Image Upload */}
          <Card className="border-2 border-red-100 hover:border-red-200 transition-colors">
            <CardHeader>
              <CardTitle className="text-red-600 flex items-center gap-2">
                <Upload className="w-5 h-5" />
                Upload Your Photo
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div
                className={`relative border-2 border-dashed rounded-lg p-6 text-center transition-colors ${
                  dragOver.person
                    ? "border-red-400 bg-red-50"
                    : personImage
                      ? "border-green-400 bg-green-50"
                      : "border-gray-300 hover:border-red-300"
                }`}
                onDrop={(e) => handleDrop(e, "person")}
                onDragOver={(e) => handleDragOver(e, "person")}
                onDragLeave={(e) => handleDragLeave(e, "person")}
              >
                {personImage ? (
                  <div className="relative">
                    <Image
                      src={personImage || "/placeholder.svg"}
                      alt="Person"
                      width={300}
                      height={400}
                      className="mx-auto rounded-lg object-cover max-h-80"
                    />
                    <Button
                      onClick={() => removeImage("person")}
                      className="absolute top-2 right-2 bg-red-500 hover:bg-red-600 text-white p-2 rounded-full"
                      size="sm"
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                ) : (
                  <div className="py-12">
                    <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-600 mb-2">Drag & drop your photo here</p>
                    <p className="text-sm text-gray-500 mb-4">or</p>
                    <Button
                      onClick={() => personInputRef.current?.click()}
                      className="bg-red-500 hover:bg-red-600 text-white"
                    >
                      Browse Files
                    </Button>
                  </div>
                )}
                <input
                  ref={personInputRef}
                  type="file"
                  accept="image/*"
                  onChange={(e) => handleImageUpload(e.target.files[0], "person")}
                  className="hidden"
                />
              </div>
              <p className="text-xs text-gray-500 mt-2">
                Best results with full-body photos, good lighting, and clear background
              </p>
            </CardContent>
          </Card>

          {/* Clothing Image Upload */}
          <Card className="border-2 border-red-100 hover:border-red-200 transition-colors">
            <CardHeader>
              <CardTitle className="text-red-600 flex items-center gap-2">
                <Upload className="w-5 h-5" />
                Upload Clothing Item
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div
                className={`relative border-2 border-dashed rounded-lg p-6 text-center transition-colors ${
                  dragOver.clothing
                    ? "border-red-400 bg-red-50"
                    : clothingImage
                      ? "border-green-400 bg-green-50"
                      : "border-gray-300 hover:border-red-300"
                }`}
                onDrop={(e) => handleDrop(e, "clothing")}
                onDragOver={(e) => handleDragOver(e, "clothing")}
                onDragLeave={(e) => handleDragLeave(e, "clothing")}
              >
                {clothingImage ? (
                  <div className="relative">
                    <Image
                      src={clothingImage || "/placeholder.svg"}
                      alt="Clothing"
                      width={300}
                      height={400}
                      className="mx-auto rounded-lg object-cover max-h-80"
                    />
                    <Button
                      onClick={() => removeImage("clothing")}
                      className="absolute top-2 right-2 bg-red-500 hover:bg-red-600 text-white p-2 rounded-full"
                      size="sm"
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                ) : (
                  <div className="py-12">
                    <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-600 mb-2">Drag & drop clothing image here</p>
                    <p className="text-sm text-gray-500 mb-4">or</p>
                    <Button
                      onClick={() => clothingInputRef.current?.click()}
                      className="bg-red-500 hover:bg-red-600 text-white"
                    >
                      Browse Files
                    </Button>
                  </div>
                )}
                <input
                  ref={clothingInputRef}
                  type="file"
                  accept="image/*"
                  onChange={(e) => handleImageUpload(e.target.files[0], "clothing")}
                  className="hidden"
                />
              </div>
              <p className="text-xs text-gray-500 mt-2">
                Upload clear images of clothing items on white/transparent background
              </p>
            </CardContent>
          </Card>

          {/* Result Display */}
          <Card className="border-2 border-red-100">
            <CardHeader>
              <CardTitle className="text-red-600 flex items-center gap-2">
                <Sparkles className="w-5 h-5" />
                Virtual Try-On Result
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center min-h-[400px] flex items-center justify-center">
                {isProcessing ? (
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-red-500 mx-auto mb-4"></div>
                    <p className="text-gray-600">Processing your virtual try-on...</p>
                    <p className="text-sm text-gray-500 mt-2">This may take a few moments</p>
                  </div>
                ) : resultImage ? (
                  <div className="relative">
                    <Image
                      src={resultImage || "/placeholder.svg"}
                      alt="Virtual Try-On Result"
                      width={300}
                      height={400}
                      className="mx-auto rounded-lg object-cover max-h-80"
                    />
                    <div className="mt-4 flex gap-2 justify-center">
                      <Button
                        onClick={downloadResult}
                        className="bg-green-500 hover:bg-green-600 text-white flex items-center gap-2"
                      >
                        <Download className="w-4 h-4" />
                        Download
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="text-center">
                    <Sparkles className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                    <p className="text-gray-500">Your virtual try-on result will appear here</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Action Buttons */}
        <div className="flex justify-center gap-4 mt-8">
          <Button
            onClick={processVirtualTryOn}
            disabled={!personImage || !clothingImage || isProcessing}
            className="bg-[#a00a11] hover:bg-red-600 text-white px-8 py-3 text-lg font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isProcessing ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                Processing...
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5 mr-2" />
                Generate Virtual Try-On
              </>
            )}
          </Button>

          <Button
            onClick={resetAll}
            variant="outline"
            className="border-red-200 text-[#a00a11] hover:bg-red-200 px-8 py-3 text-lg font-semibold bg-transparent"
          >
            <RotateCcw className="w-5 h-5 mr-2" />
            Reset All
          </Button>
        </div>
      </div>
    </div>
  )
}